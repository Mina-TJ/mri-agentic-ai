# graph.py — Builds and compiles the LangGraph StateGraph for the MRI Reservation System.

import uuid
import logging
from datetime import datetime, timezone
from functools import partial

from langgraph.graph import StateGraph, END
from openai import OpenAI

from state import MRIState
from middleware import (
    OpenAIModerationMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    HumanInTheLoopMiddleware,
)
import nodes as n

logger = logging.getLogger(__name__)


def build_graph(client: OpenAI, model: str = "gpt-4o-mini") -> StateGraph:
    moderation  = OpenAIModerationMiddleware(client)
    retry_mw    = ModelRetryMiddleware(max_retries=3, base_delay=1.0)
    tool_limit  = ToolCallLimitMiddleware(max_calls=10)
    hitl_mw     = HumanInTheLoopMiddleware()

    node_classify       = partial(n.classify,       client=client, tool_limit=tool_limit)
    node_risk_screen    = partial(n.risk_screen,    client=client, moderation=moderation)
    node_draft_response = partial(n.draft_response, client=client, retry_mw=retry_mw, tool_limit=tool_limit)
    node_human_review   = partial(n.human_review,   hitl_mw=hitl_mw)

    builder = StateGraph(MRIState)

    builder.add_node("classify",         node_classify)
    builder.add_node("risk_screen",      node_risk_screen)
    builder.add_node("escalate",         n.escalate)
    builder.add_node("identity_gate",    n.identity_gate)
    builder.add_node("retrieve_context", n.retrieve_context)
    builder.add_node("policy_gate",      n.policy_gate)
    builder.add_node("draft_response",   node_draft_response)
    builder.add_node("human_review",     node_human_review)
    builder.add_node("finalize",         n.finalize)
    builder.add_node("emit_outputs",     n.emit_outputs)

    builder.set_entry_point("classify")
    builder.add_edge("classify", "risk_screen")
    builder.add_edge("retrieve_context", "policy_gate")
    builder.add_edge("finalize", "emit_outputs")
    builder.add_edge("escalate", "emit_outputs")
    builder.add_edge("emit_outputs", END)

    def route_risk(state: MRIState) -> str:
        if state.get("terminal_status") == "ESCALATE":
            return "escalate"
        return "identity_gate"

    builder.add_conditional_edges("risk_screen", route_risk, {
        "escalate":      "escalate",
        "identity_gate": "identity_gate",
    })

    def route_identity(state: MRIState) -> str:
        status = state.get("terminal_status")
        if status == "NEED_INFO":
            return "emit_outputs"
        if status == "ESCALATE":
            return "escalate"
        return "retrieve_context"

    builder.add_conditional_edges("identity_gate", route_identity, {
        "emit_outputs":     "emit_outputs",
        "escalate":         "escalate",
        "retrieve_context": "retrieve_context",
    })

    builder.add_edge("policy_gate", "draft_response")

    def route_draft(state: MRIState) -> str:
        if state["intent"] == "prep_info" and not state.get("hitl_reason"):
            return "emit_outputs_prep"
        return "human_review"

    builder.add_conditional_edges("draft_response", route_draft, {
        "emit_outputs_prep": "emit_outputs",
        "human_review":      "human_review",
    })

    def route_hitl(state: MRIState) -> str:
        if state.get("hitl_action") == "reject":
            return "draft_response"
        return "finalize"

    builder.add_conditional_edges("human_review", route_hitl, {
        "draft_response": "draft_response",
        "finalize":       "finalize",
    })

    return builder.compile()


def init_state(user_message: str, model: str = "gpt-4o-mini", tone: str = "neutral") -> MRIState:
    return MRIState(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        user_message=user_message,

        tone=tone,
        need_fields=[],
        first_name_input=None,
        last_name_input=None,
        dob_input=None,
        scan_type_input=None,

        intent="",
        booking_id=None,
        is_urgent=False,

        identity_verified=False,

        patient_record=None,
        hours_until_appt=None,
        context_notes="",

        policy_flags=[],
        hitl_reason=None,

        draft_response="",
        hitl_action="",
        final_response="",

        terminal_status="",
        path_taken=[],
        escalation_reason=None,

        _model=model,
    )