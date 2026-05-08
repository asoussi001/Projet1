"""
Pipeline d'ingestion de documents : OCR → Chunking → Embedding → Qdrant.

Supporte : PDF natifs, PDFs scannés (OCR), Word (.docx).
"""

from __future__ import annotations

import hashlib
import io
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    doc_id: str
    filename: str
    chunk_count: int
    collection: str
    checksum: str
    already_indexed: bool = False


class DocumentIngestionPipeline:
    """
    Pipeline complet d'ingestion de documents dans Qdrant.

    Étapes :
    1. Extraction texte (Unstructured.io + OCR Tesseract si scan)
    2. Nettoyage et normalisation
    3. Chunking récursif avec overlap
    4. Embedding (multilingual-e5-large local)
    5. Upsert Qdrant avec metadata
    6. Enregistrement audit dans PostgreSQL
    """

    # Types MIME supportés
    SUPPORTED_MIME_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/plain",
        "text/markdown",
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._qdrant_client = None
        self._embedding_model = None
        self._db_session = None

    async def initialize(self) -> None:
        """Initialise les connexions (Qdrant, embedding model, DB)."""
        await self._init_qdrant()
        await self._init_embedding_model()
        logger.info("Pipeline d'ingestion initialisé.")

    async def _init_qdrant(self) -> None:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        cfg = self._settings.qdrant
        kwargs: dict[str, Any] = {"url": cfg.url}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key.get_secret_value()

        self._qdrant_client = AsyncQdrantClient(**kwargs)

        # Créer les collections si elles n'existent pas
        for collection_name in [
            cfg.collection_architecture,
            cfg.collection_governance,
            cfg.collection_deliverables,
        ]:
            existing = await self._qdrant_client.get_collections()
            names = [c.name for c in existing.collections]
            if collection_name not in names:
                await self._qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self._settings.embedding.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Collection Qdrant créée : %s", collection_name)

    async def _init_embedding_model(self) -> None:
        cfg = self._settings.embedding
        if cfg.provider.value == "local":
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(cfg.model)
            logger.info("Modèle d'embedding local chargé : %s", cfg.model)
        else:
            from langchain_openai import OpenAIEmbeddings
            self._embedding_model = OpenAIEmbeddings(model=cfg.model)
            logger.info("Modèle d'embedding OpenAI configuré : %s", cfg.model)

    # -------------------------------------------------------------------------
    # Extraction de texte
    # -------------------------------------------------------------------------

    def _extract_text(self, content: bytes, filename: str) -> str:
        """Extrait le texte brut d'un document en utilisant Unstructured.io."""
        mime_type, _ = mimetypes.guess_type(filename)

        try:
            from unstructured.partition.auto import partition
            from unstructured.staging.base import convert_to_text

            elements = partition(
                file=io.BytesIO(content),
                metadata_filename=filename,
                strategy="hi_res" if mime_type == "application/pdf" else "fast",
            )
            text = convert_to_text(elements)
            logger.debug("Extraction OK : %s (%d chars)", filename, len(text))
            return text
        except ImportError:
            logger.warning("unstructured non installé, fallback PyPDF2.")
            return self._fallback_pdf_extract(content)

    def _fallback_pdf_extract(self, content: bytes) -> str:
        """Fallback extraction PDF via PyPDF2 si Unstructured n'est pas disponible."""
        try:
            import PyPDF2

            reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            logger.error("Impossible d'extraire le texte du PDF : %s", exc)
            return ""

    # -------------------------------------------------------------------------
    # Chunking
    # -------------------------------------------------------------------------

    def _chunk_text(self, text: str, doc_id: str, metadata: dict) -> list[DocumentChunk]:
        """
        Découpe le texte en chunks avec overlap.
        Utilise RecursiveCharacterTextSplitter de LangChain pour respecter
        les frontières naturelles (paragraphes > phrases > mots).
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        cfg = self._settings.rag
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        texts = splitter.split_text(text)
        chunks = []
        for i, chunk_text in enumerate(texts):
            if chunk_text.strip():
                chunks.append(
                    DocumentChunk(
                        text=chunk_text.strip(),
                        doc_id=doc_id,
                        chunk_index=i,
                        metadata={**metadata, "chunk_index": i, "total_chunks": len(texts)},
                    )
                )
        logger.debug("Chunking : %d chunks générés pour %s", len(chunks), doc_id)
        return chunks

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def _embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        """Génère les vecteurs d'embedding pour une liste de chunks."""
        texts = [c.text for c in chunks]

        if hasattr(self._embedding_model, "encode"):
            # SentenceTransformer local
            vectors = self._embedding_model.encode(
                texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
            )
            return vectors.tolist()
        else:
            # LangChain embedding (OpenAI)
            return self._embedding_model.embed_documents(texts)

    # -------------------------------------------------------------------------
    # Upsert Qdrant
    # -------------------------------------------------------------------------

    async def _upsert_to_qdrant(
        self,
        collection: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=f"{chunk.doc_id}_{chunk.chunk_index}",
                vector=vector,
                payload={
                    "text": chunk.text,
                    "doc_id": chunk.doc_id,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        await self._qdrant_client.upsert(collection_name=collection, points=points)
        logger.debug("Upsert Qdrant : %d points dans %s", len(points), collection)

    # -------------------------------------------------------------------------
    # Point d'entrée principal
    # -------------------------------------------------------------------------

    async def ingest_document(
        self,
        content: bytes,
        filename: str,
        collection: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """
        Ingère un document complet dans Qdrant.

        Args:
            content: Contenu binaire du fichier
            filename: Nom du fichier (pour détecter le type MIME)
            collection: Collection Qdrant cible
            metadata: Métadonnées supplémentaires (project_key, doc_type, version…)
        """
        checksum = hashlib.sha256(content).hexdigest()
        doc_id = checksum[:16]
        meta = metadata or {}

        logger.info("Ingestion document : %s (checksum=%s)", filename, checksum)

        text = self._extract_text(content, filename)
        if not text.strip():
            logger.warning("Aucun texte extrait de %s — document ignoré.", filename)
            return IngestionResult(
                doc_id=doc_id,
                filename=filename,
                chunk_count=0,
                collection=collection,
                checksum=checksum,
            )

        chunks = self._chunk_text(text, doc_id, {**meta, "filename": filename, "checksum": checksum})
        vectors = self._embed_chunks(chunks)
        await self._upsert_to_qdrant(collection, chunks, vectors)

        logger.info(
            "Document ingéré : %s → %d chunks dans '%s'", filename, len(chunks), collection
        )
        return IngestionResult(
            doc_id=doc_id,
            filename=filename,
            chunk_count=len(chunks),
            collection=collection,
            checksum=checksum,
        )

    async def ingest_from_path(
        self,
        path: Path,
        collection: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        content = path.read_bytes()
        return await self.ingest_document(content, path.name, collection, metadata)
