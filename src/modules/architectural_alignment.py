"""
Module d'Alignement Architectural.

Croise les Epics/Stories Jira avec les référentiels d'architecture (DAT, ADR, LLD/HLD)
pour détecter les écarts et produire un rapport de conformité architecturale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.config.settings import Settings
from src.connectors.jira_connector import JiraConnector
from src.connectors.rovo_connector import RovoConnector
from src.pipeline.rag_pipeline import RAGPipeline, RAGResult

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    GREEN = "VERT"
    AMBER = "AMBER"
    RED = "ROUGE"


@dataclass
class AlignmentReport:
    issue_key: str
    issue_summary: str
    risk_level: RiskLevel
    compliance_score: int
    aligned_standards: list[str]
    gaps: list[dict]
    recommendations: list[str]
    summary: str
    retrieved_references: list[dict]
    tokens_used: Optional[int]
    raw_llm_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_key": self.issue_key,
            "issue_summary": self.issue_summary,
            "risk_level": self.risk_level.value,
            "compliance_score": self.compliance_score,
            "aligned_standards": self.aligned_standards,
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "summary": self.summary,
            "retrieved_references": self.retrieved_references,
            "tokens_used": self.tokens_used,
        }


class ArchitecturalAlignmentModule:
    """
    Vérifie si un Epic ou une Story Jira respecte les standards d'architecture.

    Processus :
    1. Récupération du ticket Jira complet (titre, description, champs custom)
    2. Enrichissement via Rovo (tickets liés sémantiquement, expertise)
    3. Retrieval RAG dans les collections architecture + governance
    4. Analyse LLM avec scoring et identification des gaps
    5. Structuration du rapport
    """

    def __init__(
        self,
        settings: Settings,
        jira: JiraConnector,
        rag: RAGPipeline,
        rovo: RovoConnector | None = None,
    ) -> None:
        self._settings = settings
        self._jira = jira
        self._rag = rag
        self._rovo = rovo

    async def analyze_issue(
        self,
        issue_key: str,
        include_attachments: bool = True,
    ) -> AlignmentReport:
        """
        Analyse complète de l'alignement architectural d'un ticket Jira.

        Args:
            issue_key: Clé du ticket Jira (ex: ARCH-123)
            include_attachments: Si True, analyse aussi les pièces jointes PDF
        """
        logger.info("Démarrage analyse alignement pour %s", issue_key)

        issue = await self._jira.get_issue(
            issue_key,
            fields=[
                "summary", "description", "issuetype", "status", "priority",
                "labels", "components", "customfield_10201", "customfield_10202",
            ],
        )

        rovo_context = await self._enrich_with_rovo(issue_key)

        rag_result = await self._rag.analyze_architectural_alignment(
            issue_data=issue,
            project_filter=issue_key.split("-")[0],
        )

        attachment_findings = []
        if include_attachments:
            attachment_findings = await self._analyze_attachments(issue_key)

        return self._build_report(issue, rag_result, rovo_context, attachment_findings)

    async def analyze_project_epics(self, project_key: str) -> list[AlignmentReport]:
        """Analyse tous les Epics actifs d'un projet."""
        epics = await self._jira.get_project_epics(project_key)
        reports = []
        for epic in epics:
            try:
                report = await self.analyze_issue(epic["key"], include_attachments=False)
                reports.append(report)
            except Exception as exc:
                logger.error("Erreur analyse Epic %s : %s", epic["key"], exc)
        return reports

    async def _enrich_with_rovo(self, issue_key: str) -> dict:
        """Enrichit l'analyse avec le graphe de connaissances Rovo."""
        if not self._rovo or not self._rovo.is_available:
            return {}

        related = await self._rovo.search_related_tickets(issue_key)
        return {"related_issues": related[:5]}

    async def _analyze_attachments(self, issue_key: str) -> list[dict]:
        """Analyse les pièces jointes PDF d'un ticket pour extraction de contexte."""
        attachments = await self._jira.get_attachments(issue_key)
        pdf_attachments = [
            a for a in attachments
            if a.get("mimeType") == "application/pdf"
        ]

        findings = []
        for attachment in pdf_attachments[:3]:
            try:
                content = await self._jira.download_attachment(attachment["id"])
                report = await self._rag.generate_compliance_report(
                    deliverable_text=self._extract_text_from_bytes(content),
                    deliverable_name=attachment.get("filename", "unknown.pdf"),
                    project_key=issue_key.split("-")[0],
                )
                findings.append({
                    "filename": attachment.get("filename"),
                    "compliance_report": self._parse_json_response(report.answer),
                })
            except Exception as exc:
                logger.warning("Impossible d'analyser la pièce jointe %s : %s", attachment.get("filename"), exc)

        return findings

    def _build_report(
        self,
        issue: dict,
        rag_result: RAGResult,
        rovo_context: dict,
        attachment_findings: list[dict],
    ) -> AlignmentReport:
        parsed = self._parse_json_response(rag_result.answer)
        fields = issue.get("fields", {})

        risk_map = {
            "VERT": RiskLevel.GREEN,
            "AMBER": RiskLevel.AMBER,
            "ROUGE": RiskLevel.RED,
        }
        risk_level = risk_map.get(parsed.get("risk_level", "AMBER"), RiskLevel.AMBER)

        references = [
            {
                "source": chunk.metadata.get("filename", "?"),
                "score": round(chunk.score, 3),
                "doc_type": chunk.metadata.get("doc_type", "?"),
            }
            for chunk in rag_result.retrieved_chunks
        ]

        return AlignmentReport(
            issue_key=issue.get("key", "?"),
            issue_summary=fields.get("summary", ""),
            risk_level=risk_level,
            compliance_score=parsed.get("compliance_score", 0),
            aligned_standards=parsed.get("aligned_standards", []),
            gaps=parsed.get("gaps", []),
            recommendations=parsed.get("recommendations", []),
            summary=parsed.get("summary", rag_result.answer[:500]),
            retrieved_references=references,
            tokens_used=rag_result.tokens_used,
            raw_llm_response=rag_result.answer,
        )

    @staticmethod
    def _parse_json_response(response: str) -> dict:
        """Extrait le JSON d'une réponse LLM (qui peut contenir du texte autour)."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            logger.warning("Impossible de parser la réponse JSON du LLM.")
        return {}

    @staticmethod
    def _extract_text_from_bytes(content: bytes) -> str:
        """Extraction texte rapide depuis un PDF en mémoire."""
        try:
            import io
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
