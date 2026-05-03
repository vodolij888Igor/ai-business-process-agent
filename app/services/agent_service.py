"""
Business process agent: calls OpenAI to produce a structured process plan.

Environment: OPENAI_API_KEY (required). Optional OPENAI_MODEL defaults to gpt-4o-mini.
Loads variables from a local `.env` via python-dotenv when the process starts.
"""

from __future__ import annotations

import os
from textwrap import dedent

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError

from app.schemas.agent_schema import RunBusinessAgentRequest, RunBusinessAgentResponse

# Load .env before reading OPENAI_API_KEY (safe to call multiple times).
load_dotenv()


class AgentConfigurationError(Exception):
    """Raised when the service cannot run due to missing configuration (maps to HTTP 503)."""

    def __init__(self, message: str = "OPENAI_API_KEY is not configured.") -> None:
        self.message = message
        super().__init__(message)


class AgentProviderError(Exception):
    """Raised when the OpenAI API request fails or returns unusable output (maps to HTTP 502)."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        self.message = message
        self.cause = cause
        super().__init__(message)


_PROCESS_TYPES = (
    "customer_operations",
    "sales_process",
    "support_workflow",
    "document_processing",
    "crm_follow_up",
    "internal_operations",
    "general_business_process",
)

_TOOL_IDS = (
    "email_automation",
    "lead_scoring",
    "customer_support_bot",
    "crm_follow_up",
    "document_analyzer",
    "google_sheets_automation",
    "business_process_agent",
)


def _require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if key is None or not str(key).strip():
        raise AgentConfigurationError(
            "OPENAI_API_KEY is missing. Set it in your environment or in a `.env` file."
        )
    return str(key).strip()


def _build_user_message(payload: RunBusinessAgentRequest) -> str:
    data_section = (
        "\n".join(f"- {item}" for item in payload.data_items)
        if payload.data_items
        else "(none provided)"
    )
    return dedent(
        f"""
        Analyze the following and produce the structured business process plan.

        Business task:
        {payload.business_task.strip()}

        Business context:
        {payload.business_context.strip()}

        Data items (optional):
        {data_section}

        Preferred tone for draft_response: {payload.preferred_tone}
        """
    ).strip()


def _system_prompt() -> str:
    types_list = ", ".join(_PROCESS_TYPES)
    tools_list = ", ".join(_TOOL_IDS)
    return dedent(
        f"""
        You are an expert business process analyst. Given a business task, context,
        optional data snippets, and a preferred tone, you output a concise, actionable plan.

        Rules:
        - detected_process_type MUST be exactly one of: {types_list}
        - priority MUST be exactly one of: low, medium, high
        - suggested_tools: choose 3–7 distinct items from this allowed set (use only these ids):
          {tools_list}
        - recommended_workflow: 4–8 clear, ordered steps
        - next_steps: concrete follow-ups; reference data_items when relevant
        - draft_response: stakeholder-ready narrative matching the preferred tone
        - reasoning: short justification for priority and process type
        - task_summary: one or two sentences restating the task
        """
    ).strip()


def run_business_agent(payload: RunBusinessAgentRequest) -> RunBusinessAgentResponse:
    """
    Call OpenAI with structured output matching RunBusinessAgentResponse.

    Raises:
        AgentConfigurationError: When OPENAI_API_KEY is not set.
        AgentProviderError: When OpenAI returns an error or empty structured output.
    """
    api_key = _require_api_key()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    client = OpenAI(api_key=api_key)
    user_message = _build_user_message(payload)

    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": user_message},
            ],
            response_format=RunBusinessAgentResponse,
        )
    except AuthenticationError as exc:
        raise AgentProviderError(
            "OpenAI authentication failed. Check OPENAI_API_KEY.",
            cause=exc,
        ) from exc
    except RateLimitError as exc:
        raise AgentProviderError(
            "OpenAI rate limit exceeded. Retry later.",
            cause=exc,
        ) from exc
    except APIConnectionError as exc:
        raise AgentProviderError(
            "Could not reach OpenAI. Check network connectivity.",
            cause=exc,
        ) from exc
    except APIError as exc:
        raise AgentProviderError(
            f"OpenAI API error: {exc}",
            cause=exc,
        ) from exc

    choice = completion.choices[0]
    message = choice.message
    if getattr(message, "refusal", None):
        raise AgentProviderError(
            "OpenAI refused to generate a plan for this input.",
        )
    parsed = message.parsed
    if parsed is None:
        raise AgentProviderError(
            "OpenAI returned no structured output. Try again or simplify the request.",
        )
    return parsed
