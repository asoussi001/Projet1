"""
Module de Gouvernance et Remédiation.

Analyse les tickets de dette technique et failles de sécurité,
les confronte aux notes de service internes (COBIT, ITIL, ISO 27001)
et produit des plans de remédiation priorisés.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.config.settings import Settings
from src.connectors.jira_connector import JiraConnector
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class OverallRisk(str, Enum):
    CRITICAL = "CRITIQUE"
    HIGH = "HAUTE"
    MEDIUM = "MOYENNE"
    LOW = "FAIBLE"


class RemediationPriority(str, Enum):
    P1 = "P1"  # < 1 semaine — faille sécurité ou blocant prod
    P2 = "P2"  # < 1 mois — dette impactant performance/livraison
    P3 = "P3"  # < 1 trimestre — amélioration qualitative


@dataclass
class RemediationItem:
    issue_key: str
    priority: RemediationPriority
    effort_days: int
    action: str
    rationale: str
    dependencies: list[str]


@dataclass
class GovernanceReport:
    project_key: Optional[str]
    overall_risk: OverallRisk
    total_debt_issues: int
    remediation_plan: list[RemediationItem]
    quick_wins: list[str]
    executive_summary: str
    tokens_used: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_key": self.project_key,
            "overall_risk": self.overall_risk.value,
            "total_debt_issues": self.total_debt_issues,
            "remediation_plan": [
                {
                    "issue_key": r.issue_key,
                    "priority": r.priority.value,
                    "effort_days": r.effort_days,
                    "action": r.action,
                    "rationale": r.rationale,
                    "dependencies": r.dependencies,
                }
                for r in self.remediation_plan
            ],
            "quick_wins": self.quick_wins,
            "executive_summary": self.executive_summary,
            "tokens_used": self.tokens_used,
        }

    @property
    def p1_count(self) -> int:
        return sum(1 for r in self.remediation_plan if r.priority == RemediationPriority.P1)

    @property
    def total_effort_days(self) -> int:
        return sum(r.effort_days for r in self.remediation_plan)


class GovernanceRemediationModule:
    """
    Analyse la dette technique et les failles de sécurité d'un ou plusieurs projets.

    Sources d'information :
    - Tickets Jira filtrés par labels (technical-debt, security-flaw, etc.)
    - Notes de service et directives de conformité (via RAG Qdrant)
    - Référentiels COBIT/ITIL/ISO 27001 indexés

    Sortie :
    - Rapport de gouvernance avec plan de remédiation priorisé P1/P2/P3
    - Estimation des efforts (en jours)
    - Identification des quick wins
    """

    # Labels Jira identifiant la dette technique et les failles
    DEBT_LABELS = [
        "technical-debt", "debt", "security-flaw", "vulnerability",
        "cve", "pentest-finding", "code-smell", "refactoring",
    ]

    def __init__(
        self,
        settings: Settings,
        jira: JiraConnector,
        rag: RAGPipeline,
    ) -> None:
        self._settings = settings
        self._jira = jira
        self._rag = rag

    async def analyze_project(
        self,
        project_key: str,
        max_issues: int = 50,
    ) -> GovernanceReport:
        """
        Analyse complète de la gouvernance d'un projet.

        Récupère tous les tickets de dette technique + sécurité,
        les envoie au pipeline RAG pour priorisation et plan de remédiation.
        """
        logger.info("Analyse gouvernance projet : %s", project_key)

        issues = await self._jira.get_technical_debt_issues(project_key)
        logger.info("%d tickets de dette identifiés pour %s", len(issues), project_key)

        if not issues:
            return GovernanceReport(
                project_key=project_key,
                overall_risk=OverallRisk.LOW,
                total_debt_issues=0,
                remediation_plan=[],
                quick_wins=[],
                executive_summary=f"Aucun ticket de dette technique identifié pour le projet {project_key}.",
                tokens_used=0,
            )

        context = await self._build_governance_context(project_key)
        rag_result = await self._rag.analyze_governance_remediation(
            issues=issues[:max_issues],
            governance_context=context,
        )

        return self._build_report(project_key, issues, rag_result)

    async def analyze_security_posture(self, project_key: str | None = None) -> dict:
        """
        Analyse focalisée sur les failles de sécurité uniquement.
        Retourne un score de sécurité et les CVE / findings les plus critiques.
        """
        jql_parts = [
            'labels in ("security-flaw", "vulnerability", "cve", "pentest-finding")',
            'priority in ("Critical", "High")',
        ]
        if project_key:
            jql_parts.insert(0, f"project = {project_key}")

        jql = " AND ".join(jql_parts) + " ORDER BY priority DESC, created DESC"

        issues = []
        async for issue in self._jira.search_issues(jql, max_results=20):
            issues.append(issue)

        if not issues:
            return {"security_score": 100, "critical_findings": [], "message": "Aucune faille identifiée."}

        rag_result = await self._rag.analyze_governance_remediation(
            issues=issues,
            governance_context="Focus sécurité : ISO 27001, ANSSI, OWASP Top 10",
        )

        parsed = self._parse_json(rag_result.answer)
        critical_count = sum(
            1 for r in parsed.get("remediation_plan", [])
            if r.get("priority") == "P1"
        )
        security_score = max(0, 100 - (critical_count * 20))

        return {
            "security_score": security_score,
            "total_findings": len(issues),
            "critical_p1_count": critical_count,
            "remediation_plan": parsed.get("remediation_plan", []),
            "executive_summary": parsed.get("executive_summary", ""),
        }

    async def _build_governance_context(self, project_key: str) -> str:
        """Récupère le contexte de gouvernance spécifique au projet."""
        try:
            epics = await self._jira.get_project_epics(project_key)
            epic_summaries = [e.get("fields", {}).get("summary", "") for e in epics[:5]]
            return f"Projet {project_key} — Epics : {', '.join(epic_summaries)}"
        except Exception:
            return f"Projet {project_key}"

    def _build_report(
        self,
        project_key: str,
        issues: list[dict],
        rag_result: Any,
    ) -> GovernanceReport:
        parsed = self._parse_json(rag_result.answer)

        risk_map = {
            "CRITIQUE": OverallRisk.CRITICAL,
            "HAUTE": OverallRisk.HIGH,
            "MOYENNE": OverallRisk.MEDIUM,
            "FAIBLE": OverallRisk.LOW,
        }
        overall_risk = risk_map.get(parsed.get("overall_risk", "MOYENNE"), OverallRisk.MEDIUM)

        remediation_plan = []
        for item in parsed.get("remediation_plan", []):
            prio_map = {"P1": RemediationPriority.P1, "P2": RemediationPriority.P2, "P3": RemediationPriority.P3}
            remediation_plan.append(
                RemediationItem(
                    issue_key=item.get("issue_key", "?"),
                    priority=prio_map.get(item.get("priority", "P3"), RemediationPriority.P3),
                    effort_days=item.get("effort_days", 0),
                    action=item.get("action", ""),
                    rationale=item.get("rationale", ""),
                    dependencies=item.get("dependencies", []),
                )
            )

        return GovernanceReport(
            project_key=project_key,
            overall_risk=overall_risk,
            total_debt_issues=len(issues),
            remediation_plan=remediation_plan,
            quick_wins=parsed.get("quick_wins", []),
            executive_summary=parsed.get("executive_summary", rag_result.answer[:500]),
            tokens_used=rag_result.tokens_used,
        )

    @staticmethod
    def _parse_json(response: str) -> dict:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            logger.warning("Réponse LLM non parsable en JSON.")
        return {}
