"""Tests unitaires — Module d'Alignement Architectural."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.modules.architectural_alignment import (
    ArchitecturalAlignmentModule,
    AlignmentReport,
    RiskLevel,
)
from src.pipeline.rag_pipeline import RAGResult, RetrievedChunk


SAMPLE_RAG_RESULT_GREEN = RAGResult(
    answer='{"risk_level": "VERT", "compliance_score": 92, "aligned_standards": ["OAuth 2.0", "Kong Enterprise"], "gaps": [], "recommendations": ["Aucune action requise"], "summary": "Conforme."}',
    retrieved_chunks=[
        RetrievedChunk(text="Standard Kong", score=0.95, metadata={"filename": "DAT.pdf", "doc_type": "DAT"})
    ],
    tokens_used=400,
    model_used="claude-sonnet-4-6",
)

SAMPLE_RAG_RESULT_RED = RAGResult(
    answer='{"risk_level": "ROUGE", "compliance_score": 25, "aligned_standards": [], "gaps": [{"description": "Absence OAuth", "severity": "HIGH"}], "recommendations": ["Implémenter OAuth 2.0"], "summary": "Non conforme."}',
    retrieved_chunks=[],
    tokens_used=350,
    model_used="claude-sonnet-4-6",
)

SAMPLE_ISSUE = {
    "key": "ARCH-456",
    "fields": {
        "summary": "Nouveau microservice paiement",
        "description": "Développement d'un service de paiement.",
        "issuetype": {"name": "Epic"},
        "status": {"name": "In Progress"},
        "priority": {"name": "High"},
    },
}


@pytest.fixture
def module():
    settings = MagicMock()
    jira = AsyncMock()
    rag = AsyncMock()
    rovo = AsyncMock()
    rovo.is_available = False

    jira.get_issue = AsyncMock(return_value=SAMPLE_ISSUE)
    jira.get_attachments = AsyncMock(return_value=[])

    return ArchitecturalAlignmentModule(settings, jira, rag, rovo)


class TestArchitecturalAlignmentModule:

    @pytest.mark.asyncio
    async def test_analyze_issue_green(self, module):
        module._rag.analyze_architectural_alignment = AsyncMock(return_value=SAMPLE_RAG_RESULT_GREEN)

        report = await module.analyze_issue("ARCH-456", include_attachments=False)

        assert isinstance(report, AlignmentReport)
        assert report.risk_level == RiskLevel.GREEN
        assert report.compliance_score == 92
        assert report.issue_key == "ARCH-456"
        assert len(report.gaps) == 0

    @pytest.mark.asyncio
    async def test_analyze_issue_red(self, module):
        module._rag.analyze_architectural_alignment = AsyncMock(return_value=SAMPLE_RAG_RESULT_RED)

        report = await module.analyze_issue("ARCH-456", include_attachments=False)

        assert report.risk_level == RiskLevel.RED
        assert report.compliance_score == 25
        assert len(report.gaps) == 1
        assert report.gaps[0]["severity"] == "HIGH"

    def test_parse_json_response_extracts_from_text(self, module):
        response = """
        Voici mon analyse :
        {"risk_level": "AMBER", "compliance_score": 65, "gaps": []}
        Fin de l'analyse.
        """
        result = module._parse_json_response(response)
        assert result["risk_level"] == "AMBER"
        assert result["compliance_score"] == 65

    def test_parse_json_response_returns_empty_on_failure(self, module):
        result = module._parse_json_response("Réponse non-JSON du LLM.")
        assert result == {}

    @pytest.mark.asyncio
    async def test_analyze_project_epics_handles_errors(self, module):
        """Vérifie que les erreurs sur un Epic n'interrompent pas l'analyse des autres."""
        module._jira.get_project_epics = AsyncMock(
            return_value=[
                {"key": "ARCH-1", "fields": {"summary": "Epic 1"}},
                {"key": "ARCH-2", "fields": {"summary": "Epic 2"}},
            ]
        )
        module._jira.get_issue = AsyncMock(side_effect=[
            {"key": "ARCH-1", "fields": {"summary": "Epic 1", "issuetype": {"name": "Epic"}}},
            Exception("Erreur simulée"),
        ])
        module._rag.analyze_architectural_alignment = AsyncMock(return_value=SAMPLE_RAG_RESULT_GREEN)

        reports = await module.analyze_project_epics("ARCH")
        assert len(reports) == 1
        assert reports[0].issue_key == "ARCH-1"
