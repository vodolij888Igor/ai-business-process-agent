"""
API tests for POST /run-business-agent.

OpenAI is always mocked; no network calls and no real API key required.
"""

from __future__ import annotations

import httpx
import pytest
from unittest.mock import MagicMock, patch

from openai import APIConnectionError

ALLOWED_PROCESS_TYPES = frozenset(
    {
        "customer_operations",
        "sales_process",
        "support_workflow",
        "document_processing",
        "crm_follow_up",
        "internal_operations",
        "general_business_process",
    }
)
ALLOWED_PRIORITY = frozenset({"low", "medium", "high"})

VALID_BODY = {
    "business_task": "Analyze recent customer requests and recommend next actions.",
    "business_context": "Small SaaS company with email and support channels.",
    "data_items": ["Customer A asked about pricing."],
    "preferred_tone": "professional",
}


def test_success_returns_200_and_required_fields(
    client,
    mock_openai_parse_success,
) -> None:
    response = client.post("/run-business-agent", json=VALID_BODY)
    assert response.status_code == 200
    data = response.json()

    for key in (
        "task_summary",
        "detected_process_type",
        "priority",
        "recommended_workflow",
        "suggested_tools",
        "draft_response",
        "next_steps",
        "reasoning",
    ):
        assert key in data

    mock_openai_parse_success.beta.chat.completions.parse.assert_called_once()


def test_success_response_priority_is_allowed_value(
    client,
    mock_openai_parse_success,
) -> None:
    data = client.post("/run-business-agent", json=VALID_BODY).json()
    assert data["priority"] in ALLOWED_PRIORITY


def test_success_response_detected_process_type_is_allowed_value(
    client,
    mock_openai_parse_success,
) -> None:
    data = client.post("/run-business-agent", json=VALID_BODY).json()
    assert data["detected_process_type"] in ALLOWED_PROCESS_TYPES


def test_success_suggested_tools_is_list(
    client,
    mock_openai_parse_success,
) -> None:
    data = client.post("/run-business-agent", json=VALID_BODY).json()
    assert isinstance(data["suggested_tools"], list)


def test_success_recommended_workflow_is_list(
    client,
    mock_openai_parse_success,
) -> None:
    data = client.post("/run-business-agent", json=VALID_BODY).json()
    assert isinstance(data["recommended_workflow"], list)


def test_success_next_steps_is_list(
    client,
    mock_openai_parse_success,
) -> None:
    data = client.post("/run-business-agent", json=VALID_BODY).json()
    assert isinstance(data["next_steps"], list)


def test_validation_missing_required_fields_returns_422(client) -> None:
    response = client.post("/run-business-agent", json={})
    assert response.status_code == 422


def test_validation_empty_business_task_returns_422(client) -> None:
    response = client.post(
        "/run-business-agent",
        json={"business_task": "", "business_context": "context"},
    )
    assert response.status_code == 422


def test_validation_invalid_preferred_tone_returns_422(client) -> None:
    bad = {**VALID_BODY, "preferred_tone": "invalid_tone"}
    response = client.post("/run-business-agent", json=bad)
    assert response.status_code == 422


def test_missing_openai_api_key_returns_503(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/run-business-agent", json=VALID_BODY)
    assert response.status_code == 503
    assert "detail" in response.json()


def test_openai_api_failure_returns_502(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    exc = APIConnectionError(message="Simulated OpenAI failure.", request=req)

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = exc

    with patch("app.services.agent_service.OpenAI", return_value=mock_client):
        response = client.post("/run-business-agent", json=VALID_BODY)

    assert response.status_code == 502
    assert "detail" in response.json()
