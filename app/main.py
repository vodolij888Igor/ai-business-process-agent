"""
FastAPI entrypoint for the AI Business Process Agent API.

Validates requests, runs OpenAI-backed agent logic, and maps service failures to HTTP errors.
"""

from fastapi import FastAPI, HTTPException

from app.schemas.agent_schema import RunBusinessAgentRequest, RunBusinessAgentResponse
from app.services.agent_service import (
    AgentConfigurationError,
    AgentProviderError,
    run_business_agent,
)

app = FastAPI(
    title="AI Business Process Agent",
    description=(
        "Accepts a business task, context, and optional data snippets; "
        "returns a structured, AI-ready business process plan powered by OpenAI."
    ),
    version="0.2.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe for orchestrators and quick manual checks."""
    return {"status": "ok"}


@app.post(
    "/run-business-agent",
    response_model=RunBusinessAgentResponse,
    responses={
        422: {"description": "Validation error — request body did not match the schema."},
        502: {"description": "Bad gateway — OpenAI request failed or returned unusable output."},
        503: {"description": "Service unavailable — OpenAI is not configured (missing API key)."},
    },
)
def run_business_agent_endpoint(
    body: RunBusinessAgentRequest,
) -> RunBusinessAgentResponse:
    """
    Run the business process agent and return a structured plan.

    Request fields are validated by Pydantic before this handler runs. The response
    shape is stable for downstream automation and integrations.
    """
    try:
        return run_business_agent(body)
    except AgentConfigurationError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    except AgentProviderError as exc:
        raise HTTPException(status_code=502, detail=exc.message) from exc
