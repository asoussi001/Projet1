"""Tests unitaires — RAG Pipeline."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline.rag_pipeline import RAGPipeline, RetrievedChunk, RAGResult


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.qdrant.url = "http://localhost:6333"
    settings.qdrant.api_key = None
    settings.qdrant.collection_architecture = "architecture_docs"
    settings.qdrant.collection_governance = "governance_docs"
    settings.embedding.provider = MagicMock(value="local")
    settings.embedding.model = "all-MiniLM-L6-v2"
    settings.llm.llm_provider = MagicMock()
    settings.llm.llm_provider = MagicMock()
    settings.rag.top_k = 5
    settings.rag.score_threshold = 0.5
    return settings


@pytest.fixture
def pipeline(mock_settings):
    return RAGPipeline(mock_settings)


class TestRAGPipeline:

    def test_extract_description_text_from_adf(self, pipeline):
        """Test extraction depuis le format Atlassian Document Format."""
        adf = {
            "content": [
                {
                    "content": [
                        {"type": "text", "text": "Première phrase."},
                        {"type": "text", "text": " Deuxième phrase."},
                    ]
                }
            ]
        }
        result = pipeline._extract_description_text(adf)
        assert "Première phrase." in result
        assert "Deuxième phrase." in result

    def test_extract_description_text_from_string(self, pipeline):
        result = pipeline._extract_description_text("Texte simple")
        assert result == "Texte simple"

    def test_extract_description_text_none(self, pipeline):
        result = pipeline._extract_description_text(None)
        assert result == ""

    def test_build_context_formats_chunks(self, pipeline):
        chunks = [
            RetrievedChunk(
                text="Contenu du chunk 1",
                score=0.95,
                metadata={"filename": "DAT-001.pdf", "doc_type": "DAT"},
            ),
            RetrievedChunk(
                text="Contenu du chunk 2",
                score=0.82,
                metadata={"filename": "ADR-015.pdf", "doc_type": "ADR"},
            ),
        ]
        context = pipeline._build_context(chunks)
        assert "[Ref 1]" in context
        assert "DAT-001.pdf" in context
        assert "0.95" in context
        assert "Contenu du chunk 1" in context

    @pytest.mark.asyncio
    async def test_analyze_architectural_alignment_parses_json(self, pipeline):
        """Vérifie que le pipeline parse correctement la réponse JSON du LLM."""
        mock_chunks = [
            RetrievedChunk(text="Standard API Kong", score=0.9, metadata={"filename": "DAT.pdf"})
        ]

        pipeline.retrieve = AsyncMock(return_value=mock_chunks)
        pipeline._settings.llm.llm_provider = MagicMock()

        expected_response = json.dumps({
            "risk_level": "VERT",
            "compliance_score": 90,
            "aligned_standards": ["OAuth 2.0", "TLS 1.3"],
            "gaps": [],
            "recommendations": ["Aucune action requise"],
            "summary": "Excellent alignement architectural.",
        })

        pipeline._invoke_llm = AsyncMock(return_value=(expected_response, 500))

        issue = {
            "key": "ARCH-100",
            "fields": {
                "summary": "Migration API Gateway",
                "description": "Texte de description",
                "issuetype": {"name": "Epic"},
            },
        }

        result = await pipeline.analyze_architectural_alignment(issue)
        assert isinstance(result, RAGResult)
        assert result.tokens_used == 500
        assert "VERT" in result.answer

    @pytest.mark.asyncio
    async def test_generate_compliance_report_structure(self, pipeline):
        pipeline.retrieve = AsyncMock(return_value=[])
        pipeline._invoke_llm = AsyncMock(
            return_value=(
                json.dumps({
                    "compliance_score": 75,
                    "overall_verdict": "PARTIELLEMENT_CONFORME",
                    "non_conformities": [],
                    "recommendations": [],
                }),
                300,
            )
        )

        result = await pipeline.generate_compliance_report(
            deliverable_text="Contenu du livrable",
            deliverable_name="DAT-MIGCLOUD-v1.pdf",
            project_key="MIGCLOUD",
        )
        assert result.tokens_used == 300
        assert "compliance_score" in result.answer


class TestDocumentIngestionPipeline:

    def test_chunk_text_creates_overlapping_chunks(self):
        from src.pipeline.document_ingestion import DocumentIngestionPipeline

        settings = MagicMock()
        settings.rag.chunk_size = 100
        settings.rag.chunk_overlap = 20
        settings.embedding.dimension = 384

        pipeline = DocumentIngestionPipeline(settings)
        long_text = " ".join([f"Mot{i}" for i in range(200)])

        chunks = pipeline._chunk_text(long_text, "doc-001", {"project_key": "TEST"})

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.doc_id == "doc-001"
            assert "project_key" in chunk.metadata
