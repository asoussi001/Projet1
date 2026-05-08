"""Connecteur Rovo — Interface avec le graphe de connaissances Atlassian."""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config.settings import RovoSettings

logger = logging.getLogger(__name__)


class RovoUnavailableError(Exception):
    """API Rovo indisponible (Beta instability)."""


class RovoConnector:
    """
    Client asynchrone pour l'API Rovo (Atlassian Knowledge Graph).

    L'API Rovo est en Beta — ce connecteur implémente un fallback gracieux
    si Rovo est indisponible, pour ne pas bloquer les analyses principales.
    """

    def __init__(self, settings: RovoSettings) -> None:
        self._settings = settings
        self._client: Optional[httpx.AsyncClient] = None
        self._available: bool = settings.enabled

    async def __aenter__(self) -> RovoConnector:
        if self._settings.enabled and self._settings.api_token:
            token = self._settings.api_token.get_secret_value()
            self._client = httpx.AsyncClient(
                base_url=self._settings.api_base_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=30,
            )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self._client or not self._available:
            raise RovoUnavailableError("Rovo non configuré ou désactivé.")

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=16),
            retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
            reraise=True,
        ):
            with attempt:
                response = await self._client.request(method, path, **kwargs)
                if response.status_code in (500, 503):
                    self._available = False
                    raise RovoUnavailableError(f"Rovo service error: {response.status_code}")
                response.raise_for_status()
                return response.json()

    async def semantic_search(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 20,
        project_keys: list[str] | None = None,
    ) -> list[dict]:
        """
        Recherche sémantique dans le graphe de connaissances Atlassian.

        Fallback vers liste vide si Rovo est indisponible — ne jamais bloquer
        l'analyse principale en cas d'échec Rovo.
        """
        try:
            payload: dict[str, Any] = {
                "query": query,
                "sources": sources or ["jira", "confluence"],
                "limit": limit,
            }
            if project_keys:
                payload["filters"] = {"project_keys": project_keys}

            data = await self._request("POST", "/search", json=payload)
            return data.get("results", [])

        except RovoUnavailableError:
            logger.warning("Rovo indisponible — fallback vers liste vide pour la recherche.")
            return []
        except Exception as exc:
            logger.error("Erreur inattendue Rovo : %s", exc)
            return []

    async def get_project_experts(self, project_key: str) -> list[dict]:
        """Identifie les experts internes d'un projet via le graphe Rovo."""
        try:
            data = await self._request(
                "GET",
                f"/projects/{project_key}/experts",
            )
            return data.get("experts", [])
        except (RovoUnavailableError, Exception) as exc:
            logger.warning("Impossible de récupérer les experts via Rovo : %s", exc)
            return []

    async def get_project_summary(self, project_key: str) -> dict | None:
        """Retourne un résumé synthétique d'un projet via le graphe Rovo."""
        try:
            data = await self._request("GET", f"/projects/{project_key}/summary")
            return data
        except (RovoUnavailableError, Exception):
            return None

    async def search_related_tickets(
        self, issue_key: str, relationship_types: list[str] | None = None
    ) -> list[dict]:
        """Trouve les tickets sémantiquement liés à un ticket donné."""
        try:
            payload = {
                "issue_key": issue_key,
                "relationship_types": relationship_types or ["similar", "blocking", "related"],
            }
            data = await self._request("POST", "/issues/related", json=payload)
            return data.get("related_issues", [])
        except (RovoUnavailableError, Exception) as exc:
            logger.warning("Rovo related tickets indisponible : %s", exc)
            return []

    @property
    def is_available(self) -> bool:
        return self._available and self._client is not None
