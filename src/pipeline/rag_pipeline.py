"""
Pipeline RAG (Retrieval-Augmented Generation).

Orchestre la recherche sémantique dans Qdrant + génération LLM
pour l'analyse de conformité et d'alignement architectural.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from src.config.settings import LLMProvider, Settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass
class RAGResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    tokens_used: Optional[int]
    model_used: str


# ---------------------------------------------------------------------------
# Prompts système
# ---------------------------------------------------------------------------

ALIGNMENT_SYSTEM_PROMPT = """Tu es un Architecte d'Entreprise Senior expert en gouvernance IT.
Tu analyses des tickets Jira et les confrontes aux référentiels d'architecture internes.

Ton rôle :
1. Identifier si la solution décrite dans le ticket respecte les standards d'architecture
2. Détecter les écarts par rapport aux DAT, ADR et bonnes pratiques référencées
3. Proposer des corrections ou des points de vigilance
4. Évaluer le niveau de risque : VERT (conforme), AMBER (vigilance), ROUGE (non-conforme)

Sois factuel, cite les références documentaires pertinentes, reste concis.
Réponds en JSON structuré selon le schéma demandé."""

GOVERNANCE_SYSTEM_PROMPT = """Tu es un expert en gouvernance IT (COBIT, ITIL, ISO 27001).
Tu analyses la dette technique et les failles de sécurité pour proposer des plans de remédiation.

Ton rôle :
1. Évaluer l'impact et l'urgence de chaque ticket de dette technique / sécurité
2. Croiser avec les notes de service et directives internes
3. Proposer un plan de remédiation priorisé (P1/P2/P3) avec estimation d'effort
4. Identifier les dépendances entre remédiations

Réponds en JSON structuré selon le schéma demandé."""

COMPLIANCE_SYSTEM_PROMPT = """Tu es un auditeur de conformité IT.
Tu analyses des livrables projets (DAT, LLD, rapports d'audit) par rapport aux standards internes.

Vérifie :
- Couverture des exigences architecturales
- Présence des sections obligatoires (sécurité, performance, réversibilité)
- Conformité avec les patterns d'intégration approuvés
- Respect des SLA et des normes de qualité

Réponds en JSON structuré avec une note de conformité (0-100) et les points de non-conformité."""


class RAGPipeline:
    """
    Pipeline RAG unifié pour toutes les analyses du Cockpit Tech Office.

    Supporte :
    - Analyse d'alignement architectural (Epic vs DAT/ADR)
    - Analyse de gouvernance (dette technique vs notes de service)
    - Rapport de conformité (livrable PDF vs référentiels)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._qdrant_client = None
        self._embedding_model = None
        self._llm = None

    async def initialize(self) -> None:
        await self._init_qdrant()
        await self._init_embedding()
        await self._init_llm()
        logger.info("RAG Pipeline initialisé.")

    async def _init_qdrant(self) -> None:
        from qdrant_client import AsyncQdrantClient

        cfg = self._settings.qdrant
        kwargs: dict[str, Any] = {"url": cfg.url}
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key.get_secret_value()
        self._qdrant_client = AsyncQdrantClient(**kwargs)

    async def _init_embedding(self) -> None:
        cfg = self._settings.embedding
        if cfg.provider.value == "local":
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(cfg.model)
        else:
            from langchain_openai import OpenAIEmbeddings
            self._embedding_model = OpenAIEmbeddings(model=cfg.model)

    async def _init_llm(self) -> None:
        cfg = self._settings.llm
        if cfg.llm_provider == LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model=cfg.anthropic_model,
                api_key=cfg.anthropic_api_key.get_secret_value(),
                max_tokens=cfg.anthropic_max_tokens,
            )
            self._model_name = cfg.anthropic_model
        elif cfg.llm_provider == LLMProvider.AZURE_OPENAI:
            from langchain_openai import AzureChatOpenAI
            self._llm = AzureChatOpenAI(
                azure_endpoint=cfg.azure_openai_endpoint,
                api_key=cfg.azure_openai_api_key.get_secret_value(),
                azure_deployment=cfg.azure_openai_deployment,
                api_version=cfg.azure_openai_api_version,
            )
            self._model_name = cfg.azure_openai_deployment
        else:
            raise ValueError(f"LLM provider non supporté : {cfg.llm_provider}")

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        if hasattr(self._embedding_model, "encode"):
            return self._embedding_model.encode(
                [query], normalize_embeddings=True
            )[0].tolist()
        return self._embedding_model.embed_query(query)

    async def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Recherche sémantique dans Qdrant avec filtrage optionnel.

        Le filtrage metadata_filter permet de restreindre la recherche
        à un projet, un type de document, ou une version spécifique.
        """
        k = top_k or self._settings.rag.top_k
        query_vector = self._embed_query(query)

        search_kwargs: dict[str, Any] = {
            "collection_name": collection,
            "query_vector=": query_vector,
            "limit": k,
            "score_threshold": self._settings.rag.score_threshold,
            "with_payload": True,
        }

        if metadata_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in metadata_filter.items()
            ]
            search_kwargs["query_filter"] = Filter(must=conditions)

        results = await self._qdrant_client.search(**search_kwargs)
        return [
            RetrievedChunk(
                text=r.payload.get("text", ""),
                score=r.score,
                metadata={k: v for k, v in r.payload.items() if k != "text"},
            )
            for r in results
        ]

    # -------------------------------------------------------------------------
    # Génération
    # -------------------------------------------------------------------------

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Formate les chunks récupérés en contexte lisible pour le LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("filename", chunk.metadata.get("doc_id", "?"))
            parts.append(f"[Ref {i}] Source: {source} (score={chunk.score:.2f})\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    async def _invoke_llm(self, system_prompt: str, user_message: str) -> tuple[str, int]:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        response = await self._llm.ainvoke(messages)
        tokens = getattr(response, "usage_metadata", {}).get("total_tokens", 0)
        return response.content, tokens

    # -------------------------------------------------------------------------
    # Analyses métier
    # -------------------------------------------------------------------------

    async def analyze_architectural_alignment(
        self,
        issue_data: dict,
        collections: list[str] | None = None,
        project_filter: str | None = None,
    ) -> RAGResult:
        """
        Vérifie l'alignement d'un Epic/Story Jira avec les référentiels d'architecture.

        Construit une requête sémantique à partir du titre + description du ticket,
        récupère les chunks d'architecture pertinents, puis demande au LLM d'analyser
        les écarts.
        """
        cfg_qdrant = self._settings.qdrant
        target_collections = collections or [
            cfg_qdrant.collection_architecture,
            cfg_qdrant.collection_governance,
        ]

        summary = issue_data.get("fields", {}).get("summary", "")
        description = issue_data.get("fields", {}).get("description", {})
        desc_text = self._extract_description_text(description)
        query = f"{summary}\n{desc_text}"

        meta_filter = {"project_key": project_filter} if project_filter else None

        all_chunks: list[RetrievedChunk] = []
        for collection in target_collections:
            chunks = await self.retrieve(query, collection, metadata_filter=meta_filter)
            all_chunks.extend(chunks)

        all_chunks.sort(key=lambda c: c.score, reverse=True)
        top_chunks = all_chunks[: self._settings.rag.top_k]

        context = self._build_context(top_chunks)
        user_message = f"""Analyse l'alignement architectural du ticket Jira suivant :

TICKET : {issue_data.get('key', '?')}
TITRE : {summary}
DESCRIPTION : {desc_text[:2000]}
TYPE : {issue_data.get('fields', {}).get('issuetype', {}).get('name', '?')}

RÉFÉRENCES ARCHITECTURALES PERTINENTES :
{context}

Réponds en JSON avec la structure :
{{
  "risk_level": "VERT|AMBER|ROUGE",
  "compliance_score": 0-100,
  "aligned_standards": ["..."],
  "gaps": [{{"description": "...", "severity": "HIGH|MEDIUM|LOW", "reference": "..."}}],
  "recommendations": ["..."],
  "summary": "Synthèse en 2-3 phrases"
}}"""

        answer, tokens = await self._invoke_llm(ALIGNMENT_SYSTEM_PROMPT, user_message)

        return RAGResult(
            answer=answer,
            retrieved_chunks=top_chunks,
            tokens_used=tokens,
            model_used=self._model_name,
        )

    async def analyze_governance_remediation(
        self, issues: list[dict], governance_context: str | None = None
    ) -> RAGResult:
        """
        Analyse un ensemble de tickets de dette technique et propose des plans de remédiation.
        """
        collection = self._settings.qdrant.collection_governance

        combined_query = " ".join(
            issue.get("fields", {}).get("summary", "") for issue in issues[:10]
        )
        chunks = await self.retrieve(combined_query, collection)
        context = self._build_context(chunks)

        issues_summary = "\n".join(
            f"- [{i.get('key')}] {i.get('fields', {}).get('summary', '')} "
            f"(priorité: {i.get('fields', {}).get('priority', {}).get('name', '?')})"
            for i in issues
        )

        user_message = f"""Analyse ces tickets de dette technique / sécurité et propose un plan de remédiation :

TICKETS :
{issues_summary}

DIRECTIVES INTERNES PERTINENTES :
{context}

{f'CONTEXTE ADDITIONNEL : {governance_context}' if governance_context else ''}

Réponds en JSON :
{{
  "overall_risk": "CRITIQUE|HAUTE|MOYENNE|FAIBLE",
  "remediation_plan": [
    {{
      "issue_key": "...",
      "priority": "P1|P2|P3",
      "effort_days": 0,
      "action": "...",
      "rationale": "...",
      "dependencies": ["..."]
    }}
  ],
  "quick_wins": ["..."],
  "executive_summary": "..."
}}"""

        answer, tokens = await self._invoke_llm(GOVERNANCE_SYSTEM_PROMPT, user_message)

        return RAGResult(
            answer=answer,
            retrieved_chunks=chunks,
            tokens_used=tokens,
            model_used=self._model_name,
        )

    async def generate_compliance_report(
        self,
        deliverable_text: str,
        deliverable_name: str,
        project_key: str,
    ) -> RAGResult:
        """
        Génère un rapport de conformité pour un livrable projet (PDF analysé).

        Compare le contenu du livrable avec les référentiels d'architecture
        et de gouvernance pour calculer un score de conformité.
        """
        query = f"exigences conformité {project_key} architecture sécurité standards"
        arch_chunks = await self.retrieve(
            query,
            self._settings.qdrant.collection_architecture,
            metadata_filter={"project_key": project_key},
        )
        gov_chunks = await self.retrieve(query, self._settings.qdrant.collection_governance)
        all_chunks = sorted(arch_chunks + gov_chunks, key=lambda c: c.score, reverse=True)[
            : self._settings.rag.top_k
        ]
        context = self._build_context(all_chunks)

        user_message = f"""Génère un rapport de conformité pour le livrable suivant :

LIVRABLE : {deliverable_name}
PROJET : {project_key}

CONTENU DU LIVRABLE (extrait) :
{deliverable_text[:4000]}

RÉFÉRENTIELS DE CONFORMITÉ :
{context}

Réponds en JSON :
{{
  "compliance_score": 0-100,
  "deliverable_name": "{deliverable_name}",
  "project_key": "{project_key}",
  "sections_present": ["..."],
  "sections_missing": ["..."],
  "non_conformities": [
    {{"section": "...", "finding": "...", "severity": "CRITICAL|MAJOR|MINOR", "reference": "..."}}
  ],
  "positive_points": ["..."],
  "overall_verdict": "CONFORME|PARTIELLEMENT_CONFORME|NON_CONFORME",
  "recommendations": ["..."]
}}"""

        answer, tokens = await self._invoke_llm(COMPLIANCE_SYSTEM_PROMPT, user_message)

        return RAGResult(
            answer=answer,
            retrieved_chunks=all_chunks,
            tokens_used=tokens,
            model_used=self._model_name,
        )

    @staticmethod
    def _extract_description_text(description: Any) -> str:
        """Extrait le texte brut d'une description Jira (format Atlassian Document Format)."""
        if isinstance(description, str):
            return description
        if isinstance(description, dict):
            texts = []
            for block in description.get("content", []):
                for inline in block.get("content", []):
                    if inline.get("type") == "text":
                        texts.append(inline.get("text", ""))
            return " ".join(texts)
        return ""
