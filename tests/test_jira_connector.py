"""Tests unitaires — JiraConnector."""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from src.connectors.jira_connector import (
    JiraConnector,
    JiraAuthError,
    JiraNotFoundError,
    JiraRateLimitError,
)
from src.config.settings import JiraAuthType


@pytest.fixture
def jira_settings():
    from pydantic import SecretStr
    settings = MagicMock()
    settings.base_url = "https://test.atlassian.net"
    settings.email = "test@example.com"
    settings.api_token = SecretStr("test-token")
    settings.auth_type = JiraAuthType.TOKEN
    settings.request_timeout = 10
    settings.max_retries = 2
    return settings


@pytest.fixture
def connector(jira_settings):
    return JiraConnector(jira_settings, anonymize=True)


class TestJiraConnector:

    def test_anonymize_removes_personal_data(self, connector):
        issue = {
            "key": "TEST-1",
            "fields": {
                "summary": "Test issue",
                "assignee": {"accountId": "user123", "displayName": "John Doe"},
                "reporter": {"accountId": "user456", "displayName": "Jane Doe"},
                "comment": {"comments": [{"body": "Sensitive comment"}]},
            },
        }
        result = connector._anonymize_issue(issue)

        assert result["fields"]["assignee"]["displayName"] == "ANONYMIZED"
        assert result["fields"]["reporter"]["displayName"] == "ANONYMIZED"
        assert "comment" not in result["fields"]
        assert result["fields"]["assignee"]["accountId"] != "user123"

    def test_anonymize_skipped_when_disabled(self, jira_settings):
        connector = JiraConnector(jira_settings, anonymize=False)
        issue = {
            "key": "TEST-1",
            "fields": {
                "assignee": {"accountId": "user123", "displayName": "John Doe"},
            },
        }
        result = connector._anonymize_issue(issue)
        assert result["fields"]["assignee"]["displayName"] == "John Doe"

    def test_raise_for_status_401(self, connector):
        response = MagicMock(spec=httpx.Response)
        response.status_code = 401
        with pytest.raises(JiraAuthError):
            connector._raise_for_status(response)

    def test_raise_for_status_404(self, connector):
        response = MagicMock(spec=httpx.Response)
        response.status_code = 404
        response.url = "https://test.atlassian.net/rest/api/3/issue/FAKE-999"
        with pytest.raises(JiraNotFoundError):
            connector._raise_for_status(response)

    def test_raise_for_status_429(self, connector):
        response = MagicMock(spec=httpx.Response)
        response.status_code = 429
        with pytest.raises(JiraRateLimitError):
            connector._raise_for_status(response)

    def test_raise_for_status_200_passes(self, connector):
        response = MagicMock(spec=httpx.Response)
        response.status_code = 200
        connector._raise_for_status(response)  # Should not raise

    @pytest.mark.asyncio
    async def test_get_issue_calls_correct_endpoint(self, connector):
        mock_response = {
            "key": "ARCH-123",
            "fields": {
                "summary": "Test Epic",
                "assignee": {"accountId": "abc123", "displayName": "Dev"},
            },
        }

        connector._client = AsyncMock()
        connector._client.request = AsyncMock(
            return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value=mock_response),
            )
        )

        result = await connector.get_issue("ARCH-123")
        assert result["key"] == "ARCH-123"
        assert result["fields"]["assignee"]["displayName"] == "ANONYMIZED"

    @pytest.mark.asyncio
    async def test_get_attachments_filters_pdf(self, connector):
        mock_issue = {
            "fields": {
                "attachment": [
                    {"id": "1", "filename": "spec.pdf", "mimeType": "application/pdf"},
                    {"id": "2", "filename": "photo.jpg", "mimeType": "image/jpeg"},
                ]
            }
        }

        connector._client = AsyncMock()
        connector._client.request = AsyncMock(
            return_value=MagicMock(status_code=200, json=MagicMock(return_value=mock_issue))
        )

        attachments = await connector.get_attachments("ARCH-123")
        assert len(attachments) == 2
        pdf_attachments = [a for a in attachments if a["mimeType"] == "application/pdf"]
        assert len(pdf_attachments) == 1
        assert pdf_attachments[0]["filename"] == "spec.pdf"


class TestJiraConnectorSearch:

    @pytest.mark.asyncio
    async def test_search_issues_paginates(self, connector):
        """Vérifie que la pagination fonctionne correctement."""
        page1 = {
            "issues": [{"key": f"TEST-{i}", "fields": {}} for i in range(100)],
            "total": 150,
        }
        page2 = {
            "issues": [{"key": f"TEST-{i}", "fields": {}} for i in range(100, 150)],
            "total": 150,
        }

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            data = page1 if call_count == 1 else page2
            return MagicMock(status_code=200, json=MagicMock(return_value=data))

        connector._client = AsyncMock()
        connector._client.request = AsyncMock(side_effect=mock_request)

        issues = []
        async for issue in connector.search_issues("project = TEST"):
            issues.append(issue)

        assert len(issues) == 150
        assert call_count == 2
