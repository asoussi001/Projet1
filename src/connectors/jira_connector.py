"""Connecteur Jira — Abstraction des appels REST API v3 avec retry et circuit breaker."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, AsyncIterator

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import JiraAuthType, JiraSettings

logger = logging.getLogger(__name__)

SENSITIVE_FIELDS = {"assignee", "reporter", "creator", "comment"}


class JiraError(Exception):
    """Erreur générique du connecteur Jira."""


class JiraRateLimitError(JiraError):
    """429 Too Many Requests."""


class JiraAuthError(JiraError):
    """401 / 403 — Credentials invalides ou permissions insuffisantes."""


class JiraNotFoundError(JiraError):
    """404 — Ressource introuvable."""


class JiraConnector:
    """
    Client asynchrone pour l'API Jira REST v3.

    Gère l'authentification (token ou OAuth2), la pagination, le retry
    exponentiel sur les erreurs transitoires, et la pseudonymisation RGPD.
    """

    _RETRY_EXCEPTIONS = (JiraRateLimitError, httpx.NetworkError, httpx.TimeoutException)

    def __init__(self, settings: JiraSettings, anonymize: bool = True) -> None:
        self._settings = settings
        self._anonymize = anonymize
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> JiraConnector:
        await self._init_client()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def _init_client(self) -> None:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        if self._settings.auth_type == JiraAuthType.TOKEN:
            token = self._settings.api_token.get_secret_value()
            auth = (self._settings.email, token)
            self._client = httpx.AsyncClient(
                base_url=self._settings.base_url,
                auth=auth,
                headers=headers,
                timeout=self._settings.request_timeout,
            )
        else:
            # OAuth2 : récupération du bearer token
            bearer = await self._fetch_oauth2_token()
            headers["Authorization"] = f"Bearer {bearer}"
            self._client = httpx.AsyncClient(
                base_url=self._settings.base_url,
                headers=headers,
                timeout=self._settings.request_timeout,
            )

    async def _fetch_oauth2_token(self) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._settings.oauth_token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._settings.oauth_client_id,
                    "client_secret": self._settings.oauth_client_secret.get_secret_value(),
                },
            )
            response.raise_for_status()
            return response.json()["access_token"]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        assert self._client, "Client non initialisé. Utiliser comme context manager."

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(self._RETRY_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                response = await self._client.request(method, path, **kwargs)
                self._raise_for_status(response)
                return response.json()

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        if response.status_code == 401:
            raise JiraAuthError("Authentification Jira échouée — vérifier le token.")
        if response.status_code == 403:
            raise JiraAuthError("Permissions insuffisantes pour cette ressource Jira.")
        if response.status_code == 404:
            raise JiraNotFoundError(f"Ressource introuvable : {response.url}")
        if response.status_code == 429:
            raise JiraRateLimitError("Rate limit Jira atteint.")
        response.raise_for_status()

    def _anonymize_issue(self, issue: dict) -> dict:
        """Pseudonymise les champs contenant des données personnelles (RGPD)."""
        if not self._anonymize:
            return issue

        def hash_user(user: dict | None) -> dict | None:
            if not user:
                return None
            account_id = user.get("accountId", "")
            return {
                "accountId": hashlib.sha256(account_id.encode()).hexdigest()[:16],
                "displayName": "ANONYMIZED",
            }

        fields = issue.get("fields", {})
        for field in SENSITIVE_FIELDS:
            if field in fields and isinstance(fields[field], dict):
                fields[field] = hash_user(fields[field])
        fields.pop("comment", None)
        return issue

    # -------------------------------------------------------------------------
    # API publique
    # -------------------------------------------------------------------------

    async def get_issue(
        self,
        issue_key: str,
        fields: list[str] | None = None,
        expand: list[str] | None = None,
    ) -> dict:
        """Récupère un ticket Jira complet."""
        params: dict[str, Any] = {}
        if fields:
            params["fields"] = ",".join(fields)
        if expand:
            params["expand"] = ",".join(expand)

        data = await self._request("GET", f"/rest/api/3/issue/{issue_key}", params=params)
        return self._anonymize_issue(data)

    async def search_issues(
        self,
        jql: str,
        fields: list[str] | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[dict]:
        """Itère sur tous les tickets correspondant à une requête JQL (pagination auto)."""
        start_at = 0
        default_fields = fields or [
            "summary", "status", "issuetype", "priority", "assignee",
            "reporter", "created", "updated", "duedate", "labels",
            "components", "customfield_10201", "customfield_10202",
        ]

        while True:
            data = await self._request(
                "POST",
                "/rest/api/3/search",
                json={
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": min(max_results, 100),
                    "fields": default_fields,
                },
            )
            issues = data.get("issues", [])
            total = data.get("total", 0)

            for issue in issues:
                yield self._anonymize_issue(issue)

            start_at += len(issues)
            if start_at >= total or not issues:
                break

    async def get_project_epics(self, project_key: str) -> list[dict]:
        """Retourne tous les Epics d'un projet."""
        epics = []
        async for issue in self.search_issues(
            jql=f"project = {project_key} AND issuetype = Epic ORDER BY created DESC",
            fields=["summary", "status", "duedate", "customfield_10201"],
        ):
            epics.append(issue)
        return epics

    async def get_technical_debt_issues(self, project_key: str | None = None) -> list[dict]:
        """Retourne les tickets de dette technique et failles de sécurité."""
        jql_parts = ['labels in ("technical-debt", "debt", "security-flaw", "vulnerability")']
        if project_key:
            jql_parts.insert(0, f"project = {project_key}")
        jql = " AND ".join(jql_parts) + " ORDER BY priority DESC"

        issues = []
        async for issue in self.search_issues(jql=jql):
            issues.append(issue)
        return issues

    async def get_attachments(self, issue_key: str) -> list[dict]:
        """Retourne la liste des pièces jointes d'un ticket."""
        issue = await self._request(
            "GET",
            f"/rest/api/3/issue/{issue_key}",
            params={"fields": "attachment"},
        )
        return issue.get("fields", {}).get("attachment", [])

    async def download_attachment(self, attachment_id: str) -> bytes:
        """Télécharge le contenu binaire d'une pièce jointe."""
        assert self._client
        response = await self._client.get(
            f"/rest/api/3/attachment/content/{attachment_id}"
        )
        self._raise_for_status(response)
        return response.content

    async def get_issue_links(self, issue_key: str) -> list[dict]:
        """Retourne les liens de dépendance entre tickets."""
        issue = await self._request(
            "GET",
            f"/rest/api/3/issue/{issue_key}",
            params={"fields": "issuelinks"},
        )
        return issue.get("fields", {}).get("issuelinks", [])

    async def get_board_sprints(self, board_id: int, state: str = "active") -> list[dict]:
        """Retourne les sprints d'un board Jira Agile."""
        data = await self._request(
            "GET",
            f"/rest/agile/1.0/board/{board_id}/sprint",
            params={"state": state},
        )
        return data.get("values", [])

    async def get_custom_fields(self) -> dict[str, str]:
        """Retourne le mapping ID → nom des champs custom de l'instance."""
        fields_data = await self._request("GET", "/rest/api/3/field")
        return {
            f["id"]: f["name"]
            for f in fields_data
            if f["id"].startswith("customfield_")
        }
