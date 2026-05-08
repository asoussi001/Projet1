"""
Agent Conversationnel Tech Office — LangChain ReAct Agent avec outils métier.

Répond à des questions en langage naturel sur les projets, l'architecture
et la conformité en orchestrant les outils disponibles.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

from src.config.settings import LLMProvider, Settings
from src.connectors.jira_connector import JiraConnector
from src.connectors.rovo_connector import RovoConnector
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Prompt ReAct pour l'agent Tech Office
TECH_OFFICE_AGENT_PROMPT = PromptTemplate.from_template("""Tu es le Tech Office Assistant, copilote intelligent de l'Architecture d'Entreprise.
Tu aides les architectes à piloter les projets, détecter les risques et analyser la conformité.

Tu as accès aux outils suivants :
{tools}

Pour répondre, utilise le format suivant :
Thought: Je dois analyser la question et choisir le bon outil
Action: nom_de_l_outil
Action Input: l'entrée pour l'outil
Observation: résultat de l'outil
... (répéter si nécessaire)
Thought: J'ai maintenant toutes les informations pour répondre
Final Answer: Réponse structurée et synthétique

Règles :
- Toujours citer les sources (clés Jira, noms de documents)
- Utiliser le niveau de risque RAG (Rouge/Amber/Vert) dans les rapports
- Être précis sur les dates, priorités et responsables
- Proposer des actions concrètes quand pertinent

Question: {input}
{agent_scratchpad}""")


class TechOfficeAgent:
    """
    Agent conversationnel pour le Tech Office.

    Intègre :
    - Recherche Jira (JQL natif)
    - Recherche sémantique Rovo (graphe de connaissances Atlassian)
    - RAG sur les référentiels d'architecture
    - Analyse de conformité à la demande
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
        self._executor: AgentExecutor | None = None

    async def initialize(self) -> None:
        """Construit l'agent LangChain avec ses outils."""
        tools = self._build_tools()
        llm = self._build_llm()
        agent = create_react_agent(llm, tools, TECH_OFFICE_AGENT_PROMPT)
        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        logger.info("TechOfficeAgent initialisé avec %d outils.", len(tools))

    def _build_llm(self) -> Any:
        cfg = self._settings.llm
        if cfg.llm_provider == LLMProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=cfg.anthropic_model,
                api_key=cfg.anthropic_api_key.get_secret_value(),
                max_tokens=cfg.anthropic_max_tokens,
                temperature=0,
            )
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=cfg.azure_openai_endpoint,
            api_key=cfg.azure_openai_api_key.get_secret_value(),
            azure_deployment=cfg.azure_openai_deployment,
            temperature=0,
        )

    def _build_tools(self) -> list[Tool]:
        tools = [
            Tool(
                name="search_jira_projects",
                description=(
                    "Recherche des tickets Jira via JQL. "
                    "Entrée : requête JQL valide (ex: 'project = ARCH AND issuetype = Epic AND status != Done'). "
                    "Retourne les tickets correspondants avec leur statut et priorité."
                ),
                func=self._tool_search_jira,
                coroutine=self._tool_search_jira_async,
            ),
            Tool(
                name="search_architecture_docs",
                description=(
                    "Recherche sémantique dans les référentiels d'architecture (DAT, ADR, LLD, HLD). "
                    "Entrée : question ou mots-clés en langage naturel. "
                    "Retourne les passages les plus pertinents avec leur source."
                ),
                func=self._tool_search_arch_docs,
                coroutine=self._tool_search_arch_docs_async,
            ),
            Tool(
                name="get_delivery_status",
                description=(
                    "Analyse le statut de livraison d'un projet : epics en retard, DAT manquants, risques. "
                    "Entrée : clé du projet Jira (ex: 'COREBANK'). "
                    "Retourne un rapport de statut de livraison."
                ),
                func=self._tool_delivery_status,
                coroutine=self._tool_delivery_status_async,
            ),
            Tool(
                name="generate_compliance_report",
                description=(
                    "Génère un rapport de conformité pour un ticket ou projet. "
                    "Entrée : clé du ticket Jira (ex: 'ARCH-123') ou 'PROJECT:CLE_PROJET'. "
                    "Retourne le score de conformité et les non-conformités."
                ),
                func=self._tool_compliance_report,
                coroutine=self._tool_compliance_report_async,
            ),
        ]

        if self._rovo and self._rovo.is_available:
            tools.append(
                Tool(
                    name="rovo_semantic_search",
                    description=(
                        "Recherche sémantique dans le graphe de connaissances Atlassian (Jira + Confluence). "
                        "Entrée : question en langage naturel. "
                        "Retourne les ressources Atlassian les plus pertinentes."
                    ),
                    func=self._tool_rovo_search,
                    coroutine=self._tool_rovo_search_async,
                )
            )

        return tools

    # -------------------------------------------------------------------------
    # Implémentations des outils (sync wrappers pour LangChain)
    # -------------------------------------------------------------------------

    def _tool_search_jira(self, jql: str) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._tool_search_jira_async(jql))

    async def _tool_search_jira_async(self, jql: str) -> str:
        try:
            issues = []
            async for issue in self._jira.search_issues(jql, max_results=20):
                fields = issue.get("fields", {})
                issues.append({
                    "key": issue.get("key"),
                    "summary": fields.get("summary"),
                    "status": fields.get("status", {}).get("name"),
                    "priority": fields.get("priority", {}).get("name"),
                    "duedate": fields.get("duedate"),
                })
            if not issues:
                return "Aucun ticket trouvé pour cette requête JQL."
            return json.dumps(issues, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Erreur Jira : {exc}"

    def _tool_search_arch_docs(self, query: str) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._tool_search_arch_docs_async(query))

    async def _tool_search_arch_docs_async(self, query: str) -> str:
        try:
            chunks = await self._rag.retrieve(
                query, self._settings.qdrant.collection_architecture
            )
            if not chunks:
                return "Aucun document d'architecture pertinent trouvé."
            results = [
                {
                    "source": c.metadata.get("filename", "?"),
                    "score": round(c.score, 3),
                    "excerpt": c.text[:300],
                }
                for c in chunks
            ]
            return json.dumps(results, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Erreur recherche architecture : {exc}"

    def _tool_delivery_status(self, project_key: str) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._tool_delivery_status_async(project_key))

    async def _tool_delivery_status_async(self, project_key: str) -> str:
        try:
            epics = await self._jira.get_project_epics(project_key.strip())
            from datetime import date
            today = date.today().isoformat()

            overdue = [
                e for e in epics
                if e.get("fields", {}).get("duedate") and e["fields"]["duedate"] < today
                and e.get("fields", {}).get("status", {}).get("name") not in ("Done", "Closed")
            ]

            status = {
                "project_key": project_key,
                "total_epics": len(epics),
                "overdue_epics": len(overdue),
                "overdue_details": [
                    {
                        "key": e.get("key"),
                        "summary": e.get("fields", {}).get("summary"),
                        "duedate": e.get("fields", {}).get("duedate"),
                        "status": e.get("fields", {}).get("status", {}).get("name"),
                    }
                    for e in overdue
                ],
                "risk_level": "ROUGE" if len(overdue) > 3 else ("AMBER" if len(overdue) > 0 else "VERT"),
            }
            return json.dumps(status, ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Erreur statut livraison : {exc}"

    def _tool_compliance_report(self, input_str: str) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._tool_compliance_report_async(input_str))

    async def _tool_compliance_report_async(self, input_str: str) -> str:
        try:
            input_str = input_str.strip()
            if input_str.startswith("PROJECT:"):
                project_key = input_str.split(":", 1)[1].strip()
                issues = await self._jira.get_project_epics(project_key)
                if not issues:
                    return f"Aucun Epic trouvé pour le projet {project_key}."
                issue_data = issues[0]
            else:
                issue_data = await self._jira.get_issue(input_str)

            rag_result = await self._rag.analyze_architectural_alignment(issue_data)
            return rag_result.answer
        except Exception as exc:
            return f"Erreur génération rapport : {exc}"

    def _tool_rovo_search(self, query: str) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(self._tool_rovo_search_async(query))

    async def _tool_rovo_search_async(self, query: str) -> str:
        try:
            results = await self._rovo.semantic_search(query)
            if not results:
                return "Aucun résultat Rovo pour cette requête."
            return json.dumps(results[:5], ensure_ascii=False, indent=2)
        except Exception as exc:
            return f"Erreur Rovo : {exc}"

    # -------------------------------------------------------------------------
    # Interface principale
    # -------------------------------------------------------------------------

    async def ask(self, question: str) -> dict[str, Any]:
        """
        Pose une question en langage naturel à l'agent.

        Returns:
            Dict avec 'answer' (réponse finale), 'steps' (raisonnement),
            et 'tools_used' (liste des outils invoqués).
        """
        if not self._executor:
            raise RuntimeError("Agent non initialisé. Appeler initialize() d'abord.")

        logger.info("Agent query : %s", question[:100])
        result = await self._executor.ainvoke({"input": question})

        tools_used = list({
            step[0].tool
            for step in result.get("intermediate_steps", [])
        })

        return {
            "answer": result.get("output", ""),
            "tools_used": tools_used,
            "steps_count": len(result.get("intermediate_steps", [])),
        }
