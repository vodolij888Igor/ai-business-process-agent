"""
Request and response models for the business process agent endpoint.

These models enforce shape and types at the API boundary and generate OpenAPI docs.
"""

from typing import Literal

from pydantic import BaseModel, Field


class RunBusinessAgentRequest(BaseModel):
    """Input describing what to plan for and what context or raw data is available."""

    business_task: str = Field(
        ...,
        min_length=1,
        description="The business objective or question the agent should address.",
        examples=["Analyze recent customer requests and recommend the next business actions."],
    )
    business_context: str = Field(
        ...,
        min_length=1,
        description="Background about the company, domain, or constraints.",
        examples=[
            "Small SaaS company that receives customer emails, lead forms, and support requests."
        ],
    )
    data_items: list[str] = Field(
        default_factory=list,
        description="Optional unstructured snippets (emails, tickets, notes) to ground the plan.",
    )
    preferred_tone: Literal["professional", "friendly", "concise"] = Field(
        default="professional",
        description="Tone for draft narrative fields in the response.",
    )


class RunBusinessAgentResponse(BaseModel):
    """Structured business process plan suitable for downstream AI or automation tools."""

    task_summary: str = Field(..., description="Short restatement of the business task.")
    detected_process_type: str = Field(
        ...,
        description=(
            "High-level category inferred from the task (e.g. customer_operations, "
            "sales_process, support_workflow, document_processing, crm_follow_up, "
            "internal_operations, general_business_process)."
        ),
    )
    priority: Literal["low", "medium", "high"] = Field(
        ...,
        description="Suggested urgency for executing the workflow.",
    )
    recommended_workflow: list[str] = Field(
        ...,
        description="Ordered steps a team or automation could follow.",
    )
    suggested_tools: list[str] = Field(
        ...,
        description=(
            "Tool or integration ideas aligned with the workflow (labels only; "
            "e.g. email_automation, crm_follow_up, business_process_agent)."
        ),
    )
    draft_response: str = Field(
        ...,
        description="Narrative summary the business could adapt for stakeholders or customers.",
    )
    next_steps: list[str] = Field(
        ...,
        description="Concrete follow-up actions tied to inputs when possible.",
    )
    reasoning: str = Field(
        ...,
        description="Brief justification for the plan (priority, process type, and key inputs).",
    )
