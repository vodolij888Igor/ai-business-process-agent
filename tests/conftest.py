"""
Shared pytest fixtures: isolated env vars and mocked OpenAI client.

Tests never call the real OpenAI API.
"""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.agent_schema import RunBusinessAgentResponse

# Valid enum-like values matching production prompts (used for mock LLM output).
MOCK_RESPONSE = RunBusinessAgentResponse(
    task_summary="Analyze customer requests and recommend actions.",
    detected_process_type="customer_operations",
    priority="high",
    recommended_workflow=[
        "Classify each request.",
        "Prioritize by urgency.",
        "Draft responses.",
        "Assign owners.",
    ],
    suggested_tools=[
        "email_automation",
        "lead_scoring",
        "customer_support_bot",
        "crm_follow_up",
    ],
    draft_response="Recommended plan based on the inputs provided.",
    next_steps=[
        "Follow up on billing items.",
        "Schedule demos for qualified leads.",
    ],
    reasoning="Multiple customer touchpoints require triage and ownership.",
)


@pytest.fixture(autouse=True)
def _tests_use_dummy_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure POST /run-business-agent does not depend on a developer's real .env key."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-not-used-with-patch")


@pytest.fixture
def client() -> Generator:
    """FastAPI synchronous test client (uses httpx under the hood)."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_openai_parse_success() -> Generator[MagicMock, None, None]:
    """Patch OpenAI client so parse() returns a structured RunBusinessAgentResponse."""
    msg = MagicMock()
    msg.parsed = MOCK_RESPONSE
    msg.refusal = None

    choice = MagicMock()
    choice.message = msg

    completion = MagicMock()
    completion.choices = [choice]

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = completion

    with patch("app.services.agent_service.OpenAI", return_value=mock_client):
        yield mock_client
