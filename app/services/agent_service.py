"""
Business process agent logic (v1: deterministic placeholders).

This module simulates an agent by classifying inputs with simple keyword rules and
returning a consistent structured plan. Swap this implementation later for an LLM
or orchestration layer without changing the public API contract.
"""

import re
from typing import Literal

from app.schemas.agent_schema import RunBusinessAgentRequest, RunBusinessAgentResponse


def _normalize_summary(text: str) -> str:
    """Produce a short, readable one-line summary from the raw task string."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) > 200:
        return cleaned[:197] + "..."
    return cleaned


def _infer_process_type(task: str, context: str) -> str:
    """Map free text to a coarse process category (placeholder heuristic)."""
    blob = f"{task} {context}".lower()
    if any(k in blob for k in ("customer", "support", "ticket", "email", "lead", "demo")):
        return "customer_operations"
    if any(k in blob for k in ("invoice", "billing", "payment", "finance", "budget")):
        return "financial_operations"
    if any(k in blob for k in ("hire", "onboard", "hr", "employee", "payroll")):
        return "people_operations"
    if any(k in blob for k in ("inventory", "supply", "vendor", "procurement")):
        return "supply_chain"
    return "general_business_process"


def _infer_priority(task: str, data_items: list[str]) -> Literal["low", "medium", "high"]:
    """Estimate urgency from keywords in the task and optional data items."""
    urgent = ("urgent", "critical", "outage", "legal", "security", "billing issue")
    blob = " ".join([task, *data_items]).lower()
    if any(u in blob for u in urgent):
        return "high"
    if len(data_items) >= 4:
        return "high"
    if len(data_items) <= 1 and len(task) < 80:
        return "low"
    return "medium"


def _tools_for_process(process_type: str) -> list[str]:
    """Suggest integration categories aligned with the detected process (labels only)."""
    catalog: dict[str, list[str]] = {
        "customer_operations": [
            "email_automation",
            "lead_scoring",
            "customer_support_bot",
            "crm_follow_up",
        ],
        "financial_operations": [
            "invoice_automation",
            "payment_gateway",
            "erp_sync",
            "reporting_dashboard",
        ],
        "people_operations": [
            "ats_integration",
            "hris_sync",
            "document_signing",
            "onboarding_playbooks",
        ],
        "supply_chain": [
            "inventory_management",
            "vendor_portal",
            "demand_forecasting",
            "logistics_tracking",
        ],
        "general_business_process": [
            "workflow_automation",
            "notification_hub",
            "knowledge_base",
            "analytics_pipeline",
        ],
    }
    return catalog.get(process_type, catalog["general_business_process"])


def _workflow_steps(process_type: str) -> list[str]:
    """Return a generic ordered checklist appropriate for the process family."""
    if process_type == "customer_operations":
        return [
            "Classify each customer request.",
            "Prioritize billing and demo-related requests.",
            "Prepare follow-up responses.",
            "Assign next actions to the appropriate team.",
        ]
    if process_type == "financial_operations":
        return [
            "Gather supporting documents and transaction references.",
            "Validate amounts and responsible parties.",
            "Trigger approvals or corrections per policy.",
            "Record outcomes and notify stakeholders.",
        ]
    if process_type == "people_operations":
        return [
            "Confirm role requirements and compliance checks.",
            "Coordinate interviews or onboarding tasks.",
            "Update HR systems and access provisioning.",
            "Communicate timelines to candidates or employees.",
        ]
    if process_type == "supply_chain":
        return [
            "Verify stock levels and reorder thresholds.",
            "Contact vendors or carriers as needed.",
            "Update purchase orders and delivery expectations.",
            "Monitor fulfillment through closure.",
        ]
    return [
        "Clarify the objective and success criteria.",
        "Identify stakeholders and required inputs.",
        "Execute the minimal viable workflow.",
        "Review results and capture lessons learned.",
    ]


def _next_steps_from_data(data_items: list[str]) -> list[str]:
    """Derive actionable bullets from unstructured lines when possible."""
    if not data_items:
        return [
            "Collect relevant data sources (emails, CRM, tickets).",
            "Confirm ownership for each workstream.",
            "Schedule a short review with stakeholders.",
        ]

    steps: list[str] = []
    for item in data_items[:10]:
        lower = item.lower()
        customer_match = re.match(r"^(Customer\s+[A-Za-z0-9]+)\b", item.strip())
        who = customer_match.group(1) if customer_match else "the customer"

        if "billing" in lower or "invoice" in lower or "payment" in lower:
            steps.append(f"Contact {who} about the billing issue.")
        elif "demo" in lower or "pricing" in lower:
            steps.append(f"Schedule a demo with {who}.")
        elif "integrat" in lower or "api" in lower or "google sheets" in lower:
            steps.append(f"Send integration details to {who}.")
        else:
            steps.append(f"Triage and respond to {who}: {item[:80]}")
    return steps[:8]


def _draft_response(
    task_summary: str,
    tone: str,
    workflow: list[str],
    next_steps: list[str],
) -> str:
    """Compose a short narrative suitable for stakeholders (placeholder text)."""
    tone_intro = {
        "professional": "Here is a recommended action plan aligned with your objectives.",
        "friendly": "Thanks for the context — here is a practical plan your team can run with.",
        "concise": "Summary plan:",
    }
    intro = tone_intro.get(tone, tone_intro["professional"])
    bullets = "\n".join(f"- {w}" for w in workflow[:4])
    actions = "\n".join(f"- {n}" for n in next_steps[:5])
    return (
        f"{intro}\n\n"
        f"Focus: {task_summary}\n\n"
        f"Suggested workflow:\n{bullets}\n\n"
        f"Immediate next moves:\n{actions}"
    )


def _reasoning_snippet(
    process_type: str,
    priority: str,
    data_count: int,
) -> str:
    """Explain why this plan was chosen (rule-based stand-in for model rationale)."""
    data_note = "Multiple data items were provided. " if data_count else ""
    return (
        f"The task maps to '{process_type}' with suggested priority '{priority}'. "
        f"{data_note}"
        "The workflow emphasizes classification, prioritization, and clear ownership "
        "so outputs stay actionable for humans or downstream automation."
    )


def run_business_agent(payload: RunBusinessAgentRequest) -> RunBusinessAgentResponse:
    """
    Produce an AI-ready structured plan from the request.

    v1 uses only local rules—no external APIs—so responses are predictable for demos
    and integration tests. Replace internals with an LLM call when ready.
    """
    task_summary = _normalize_summary(payload.business_task)
    detected = _infer_process_type(payload.business_task, payload.business_context)
    priority = _infer_priority(payload.business_task, payload.data_items)
    workflow = _workflow_steps(detected)
    tools = _tools_for_process(detected)
    next_steps = _next_steps_from_data(payload.data_items)
    draft = _draft_response(
        task_summary,
        payload.preferred_tone,
        workflow,
        next_steps,
    )
    reasoning = _reasoning_snippet(detected, priority, len(payload.data_items))

    return RunBusinessAgentResponse(
        task_summary=task_summary,
        detected_process_type=detected,
        priority=priority,
        recommended_workflow=workflow,
        suggested_tools=tools,
        draft_response=draft,
        next_steps=next_steps,
        reasoning=reasoning,
    )
