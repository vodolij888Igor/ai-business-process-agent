"""
FastAPI entrypoint for the AI Business Process Agent API.

Exposes a single primary endpoint that validates input, runs placeholder agent logic,
and returns a structured process plan for portfolio and integration demos.
"""

from fastapi import FastAPI

from app.schemas.agent_schema import RunBusinessAgentRequest, RunBusinessAgentResponse
from app.services.agent_service import run_business_agent

app = FastAPI(
    title="AI Business Process Agent",
    description=(
        "Accepts a business task, context, and optional data snippets; "
        "returns a structured, AI-ready business process plan (v1 uses local placeholder logic)."
    ),
    version="0.1.0",
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
    },
)
def run_business_agent_endpoint(
    body: RunBusinessAgentRequest,
) -> RunBusinessAgentResponse:
    """
    Run the business process agent and return a structured plan.

    Request fields are validated by Pydantic before this handler runs. The response
    shape is stable so clients can depend on it when swapping in a real LLM later.
    """
    return run_business_agent(body)
