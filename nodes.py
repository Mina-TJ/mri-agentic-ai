"""
nodes.py — All LangGraph node functions for the MRI Reservation System.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import OpenAI

from state import MRIState
from middleware import (
    PIIMiddleware,
    OpenAIModerationMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    HumanInTheLoopMiddleware,
    FilesystemMiddleware,
)

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent / "data"
PATIENT_DB = DATA_DIR / "patient_db.json"
PREP_FILE  = DATA_DIR / "prep_instructions.json"

# ── FIX 1: Only real medical emergencies — removed "urgently", "emergency" ──
URGENT_KEYWORDS = [
    "chest pain",
    "heart attack",
    "can't breathe",
    "cannot breathe",
    "shortness of breath",
    "pressure in chest",
    "bleeding heavily",
    "stroke",
    "unconscious",
    "collapsed",
    "seizure",
    "allergic reaction",
    "anaphylaxis",
    "dying",
    "call 911",
    "call ambulance",
]


def _normalize_scan_type(text: str) -> str:
    t = (text or "").strip().lower()
    mapping = {
        "brain":     "brain_mri",
        "brain mri": "brain_mri",
        "knee":      "knee_mri",
        "knee mri":  "knee_mri",
        "spine":     "spine_mri",
        "spine mri": "spine_mri",
        "general":   "general",
    }
    return mapping.get(t, t if t else "general")


def classify(state: MRIState, client: OpenAI, tool_limit: ToolCallLimitMiddleware) -> MRIState:
    tool_limit.check(state["run_id"], "classify")
    state = {**state, "terminal_status": ""}
    logger.info("[classify] Starting intent classification.")

    msg_lower = state["user_message"].lower()
    is_urgent = any(kw in msg_lower for kw in URGENT_KEYWORDS)

    import re
    match = re.search(r"\bBK-?\d{4}\b", state["user_message"], re.IGNORECASE)
    if match:
        digits     = re.search(r"\d{4}", match.group(0)).group(0)
        booking_id = f"BK-{digits}"
    else:
        booking_id = state.get("booking_id")

    system_prompt = (
        "You are an intent classifier for an MRI reservation system.\n"
        "Classify the patient message into exactly ONE of these intents:\n"
        "  reschedule\n  cancel\n  prep_info\n  general\n\n"
        "Also assess if the message describes a REAL medical emergency "
        "(chest pain, stroke, cannot breathe, seizure, unconscious, anaphylaxis). "
        "The word 'urgent' or 'urgently' alone is NOT a medical emergency.\n"
        'Return JSON only: {"intent":"<one>", "llm_urgent": true|false}'
    )

    retry = ModelRetryMiddleware()

    def _call():
        return client.chat.completions.create(
            model=state.get("_model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": state["user_message"]},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

    response   = retry.call(_call)
    result     = json.loads(response.choices[0].message.content)
    intent     = result.get("intent", "general")
    llm_urgent = bool(result.get("llm_urgent", False))
    is_urgent  = is_urgent or llm_urgent

    path = state["path_taken"] + ["classify"]
    logger.info("[classify] intent=%s, booking_id=%s, is_urgent=%s", intent, booking_id, is_urgent)

    return {
        **state,
        "intent":     intent,
        "booking_id": booking_id,
        "is_urgent":  is_urgent,
        "path_taken": path,
    }


def risk_screen(state: MRIState, client: OpenAI, moderation: OpenAIModerationMiddleware) -> MRIState:
    logger.info("[risk_screen] Screening. is_urgent=%s", state["is_urgent"])

    mod_result = moderation.screen(state["user_message"])
    if mod_result["flagged"]:
        logger.warning("[risk_screen] Moderation flagged (logged only): %s", mod_result["categories"])

    safe_msg = PIIMiddleware.mask(state["user_message"], state.get("booking_id"))
    logger.info("[risk_screen] Screened message (masked): %s", safe_msg[:80])

    path = state["path_taken"] + ["risk_screen"]

    if state["is_urgent"]:
        logger.warning("[risk_screen] URGENT flag set → routing to escalate.")
        return {
            **state,
            "terminal_status":   "ESCALATE",
            "escalation_reason": "Medical urgency detected in patient message.",
            "path_taken":        path,
        }

    return {**state, "path_taken": path}


def escalate(state: MRIState) -> MRIState:
    reason = state.get("escalation_reason", "Escalation triggered.")
    logger.warning("[escalate] ESCALATE — Reason: %s", reason)
    path = state["path_taken"] + ["escalate"]
    return {
        **state,
        "terminal_status":   "ESCALATE",
        "escalation_reason": reason,
        "path_taken":        path,
    }


def identity_gate(state: MRIState) -> MRIState:
    intent     = state["intent"]
    path       = state["path_taken"] + ["identity_gate"]
    booking_id = state.get("booking_id")
    last_name  = (state.get("last_name_input") or "").strip()
    dob        = (state.get("dob_input") or "").strip()

    logger.info("[identity_gate] intent=%s booking_id=%s last_name?=%s dob?=%s",
                intent, booking_id, bool(last_name), bool(dob))

    # Step 0 — prep_info general path (no booking ID needed)
    if intent == "prep_info" and not booking_id:
        scan_type = state.get("scan_type_input")
        if not scan_type:
            return {
                **state,
                "terminal_status": "NEED_INFO",
                "need_fields":     ["scan_type"],
                "path_taken":      path,
            }
        return {**state, "path_taken": path}

    # Step 1 — booking_id required
    if not booking_id:
        return {
            **state,
            "terminal_status": "NEED_INFO",
            "need_fields":     ["booking_id"],
            "path_taken":      path,
        }

    # Step 2 — last name required
    if not last_name:
        return {
            **state,
            "terminal_status": "NEED_INFO",
            "need_fields":     ["last_name"],
            "path_taken":      path,
        }

    # Step 3 — DOB required
    if not dob:
        return {
            **state,
            "terminal_status": "NEED_INFO",
            "need_fields":     ["dob"],
            "path_taken":      path,
        }

    # Step 4 — verify against DB
    try:
        db  = FilesystemMiddleware.read_json(str(PATIENT_DB))
        rec = db["patients"].get(booking_id)
    except Exception as e:
        logger.error("[identity_gate] DB read error: %s", e)
        rec = None

    if rec is None:
        return {
            **state,
            "terminal_status":   "ESCALATE",
            "escalation_reason": "Booking ID not found. Possible identity mismatch.",
            "path_taken":        path,
        }

    db_last = str(rec.get("last_name", "")).strip().lower()
    db_dob  = str(rec.get("dob", "")).strip()

    if db_last != last_name.lower() or db_dob != dob:
        return {
            **state,
            "terminal_status":   "ESCALATE",
            "escalation_reason": "Identity verification failed (last name / DOB mismatch).",
            "path_taken":        path,
        }

    logger.info("[identity_gate] Identity verified for BK-XXXX.")
    return {
        **state,
        "terminal_status":   "",
        "identity_verified": True,
        "patient_record":    rec,
        "path_taken":        path,
    }


def retrieve_context(state: MRIState) -> MRIState:
    path        = state["path_taken"] + ["retrieve_context"]
    context_parts = []
    hours_until   = None
    patient_rec   = state.get("patient_record")

    scan_type = "general"
    if patient_rec:
        scan_type = patient_rec.get("appointment", {}).get("scan_type", "general")
    else:
        scan_type = _normalize_scan_type(state.get("scan_type_input") or "general")

    try:
        prep_db = FilesystemMiddleware.read_json(str(PREP_FILE))
        prep    = prep_db["prep_instructions"].get(scan_type) or \
                  prep_db["prep_instructions"]["general"]
        instr_text = "\n".join(f"  • {i}" for i in prep["instructions"])
        context_parts.append(
            f"PREP INSTRUCTIONS ({prep['scan_name']}, ~{prep['duration']}):\n{instr_text}"
        )
        if prep.get("contraindications"):
            ci = "\n".join(f"  • {c}" for c in prep["contraindications"])
            context_parts.append(f"CONTRAINDICATIONS:\n{ci}")
    except Exception as e:
        logger.error("[retrieve_context] Prep load error: %s", e)
        context_parts.append("Prep instructions unavailable.")

    if patient_rec:
        try:
            appt        = patient_rec.get("appointment", {})
            appt_dt_str = f"{appt['date']}T{appt['time']}:00"
            appt_dt     = datetime.fromisoformat(appt_dt_str).replace(tzinfo=timezone.utc)
            now         = datetime.now(timezone.utc)
            delta       = appt_dt - now
            hours_until = max(0, int(delta.total_seconds() / 3600))
            context_parts.append(
                f"APPOINTMENT: {appt['date']} at {appt['time']} — ~{hours_until}h from now. "
                f"Scan: {appt.get('scan_type', 'unknown')}."
            )
        except Exception as e:
            logger.error("[retrieve_context] Time calculation error: %s", e)

    return {
        **state,
        "context_notes":    "\n\n".join(context_parts),
        "hours_until_appt": hours_until,
        "path_taken":       path,
    }


def policy_gate(state: MRIState) -> MRIState:
    path        = state["path_taken"] + ["policy_gate"]
    intent      = state["intent"]
    flags       = []
    hitl_reason = None

    if intent == "prep_info":
        return {**state, "policy_flags": [], "hitl_reason": None, "path_taken": path}

    hours = state.get("hours_until_appt")
    if intent == "cancel" and hours is not None and hours < 24:
        flags.append("late_cancel")
        hitl_reason = f"late_cancel: appointment is {hours}h away (< 24h policy)"

    patient_rec      = state.get("patient_record") or {}
    appt             = patient_rec.get("appointment", {})
    reschedule_count = appt.get("reschedule_count", 0)
    if intent == "reschedule" and reschedule_count >= 2:
        flags.append("repeat_reschedule")
        hitl_reason = hitl_reason or \
            f"repeat_reschedule: patient has rescheduled {reschedule_count} times"

    return {**state, "policy_flags": flags, "hitl_reason": hitl_reason, "path_taken": path}


def draft_response(state: MRIState, client: OpenAI,
                   retry_mw: ModelRetryMiddleware,
                   tool_limit: ToolCallLimitMiddleware) -> MRIState:
    tool_limit.check(state["run_id"], "draft_response")
    path = state["path_taken"] + ["draft_response"]

    tone = (state.get("tone") or "neutral").lower()

    # ── FIX 2: Staff name in sign-off ────────────────────
    import os
    staff_name = os.getenv("STAFF_NAME", "Mina")

    tone_instructions = {
        "friendly": f"Use warm, friendly, empathetic language. Sign off with 'Warm regards,\\n{staff_name} — MRI Appointment Assistant'.",
        "neutral":  f"Use neutral, professional language. Sign off with 'Regards,\\n{staff_name} — MRI Appointment Assistant'.",
        "brief":    f"Be brief and direct. Keep it short. Sign off with '{staff_name}, MRI Team'.",
        "formal":   f"Use formal, detailed language. Sign off with 'Yours sincerely,\\n{staff_name} — MRI Appointment Assistant'.",
    }
    tone_line = tone_instructions.get(tone, tone_instructions["neutral"])

    # Get patient first name if available
    patient_name = state.get("first_name_input") or "the patient"

    system_prompt = (
        "You are an MRI appointment assistant.\n"
        f"Tone instructions: {tone_line}\n"
        f"Address the patient by their first name: {patient_name}\n"
        "Rules:\n"
        "- Do NOT provide clinical or medical advice.\n"
        "- Do NOT include debug or system notes.\n"
        "- Keep it concise and actionable.\n"
        "- Use the exact sign-off shown in the tone instructions — do not write [Your Name].\n"
    )

    user_content = (
        f"Patient request: {state['user_message']}\n\n"
        f"Intent: {state['intent']}\n"
        f"Context:\n{state.get('context_notes', '')}\n\n"
        f"Policy flags: {state.get('policy_flags', [])}\n"
        f"HITL reason (internal only): {state.get('hitl_reason')}\n\n"
        "Write the patient-facing response.\n"
        "If cancel <24h, mention subject to manager review.\n"
        "If prep_info, list instructions clearly."
    )

    def _call():
        return client.chat.completions.create(
            model=state.get("_model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,
            max_tokens=600,
        )

    response = retry_mw.call(_call)
    draft    = response.choices[0].message.content.strip()
    logger.info("[draft_response] Draft generated (%d chars).", len(draft))

    if state["intent"] == "prep_info" and not state.get("hitl_reason"):
        return {
            **state,
            "draft_response":  draft,
            "final_response":  draft,
            "terminal_status": "READY",
            "path_taken":      path,
        }

    return {**state, "draft_response": draft, "path_taken": path}


def human_review(state: MRIState, hitl_mw: HumanInTheLoopMiddleware) -> MRIState:
    path = state["path_taken"] + ["human_review"]
    action, final_text = hitl_mw.review(
        draft=state["draft_response"],
        hitl_reason=state.get("hitl_reason"),
    )
    return {
        **state,
        "hitl_action":    action,
        "final_response": final_text if action != "reject" else "",
        "path_taken":     path,
    }


def finalize(state: MRIState) -> MRIState:
    path       = state["path_taken"] + ["finalize"]
    intent     = state["intent"]
    booking_id = state.get("booking_id")
    patient_rec = state.get("patient_record")

    if booking_id and patient_rec:
        try:
            db   = FilesystemMiddleware.read_json(str(PATIENT_DB))
            appt = db["patients"][booking_id]["appointment"]
            if intent == "cancel":
                appt["status"] = "cancelled"
            elif intent == "reschedule":
                appt["reschedule_count"] = appt.get("reschedule_count", 0) + 1
                appt["status"] = "reschedule_pending"
            FilesystemMiddleware.write_json(str(PATIENT_DB), db)
        except Exception as e:
            logger.error("[finalize] DB write-back failed: %s", e)

    final_resp = state.get("final_response") or state.get("draft_response", "")
    return {
        **state,
        "terminal_status": "READY",
        "final_response":  final_resp,
        "path_taken":      path,
    }


def emit_outputs(state: MRIState) -> MRIState:
    path   = state["path_taken"] + ["emit_outputs"]
    status = state.get("terminal_status") or "READY"

    if status == "NEED_INFO":
        nf = state.get("need_fields", [])
        if "booking_id" in nf:
            final_resp = "Sure — please tell me your booking ID (example: BK-1001)."
        elif "last_name" in nf:
            final_resp = "Thanks. What is your last name?"
        elif "dob" in nf:
            final_resp = "Got it. What is your date of birth? (YYYY-MM-DD)"
        elif "scan_type" in nf:
            final_resp = "What type of MRI is it — brain, knee, spine, or general?"
        else:
            final_resp = "Please provide the missing details so I can continue."
    elif status == "ESCALATE":
        final_resp = (
            "If you believe this is urgent, please seek immediate care "
            "or contact local urgent/emergency support. "
            "We have flagged your case for staff follow-up."
        )
    else:
        final_resp = state.get("final_response", "")

    evidence = (
        f"\n{'═' * 60}\n"
        f"  RUN EVIDENCE\n"
        f"{'═' * 60}\n"
        f"  run_id         : {state['run_id']}\n"
        f"  timestamp      : {state['timestamp']}\n"
        f"  terminal_status: {status}\n"
        f"  path_taken     : {' → '.join(path)}\n"
        f"{'─' * 60}\n"
        f"  FINAL RESPONSE :\n\n"
        f"{final_resp}\n"
        f"{'═' * 60}\n"
    )
    print(evidence)

    log_dir  = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"trace_{state['run_id'][:8]}.json"
    trace = {
        "run_id":          state["run_id"],
        "timestamp":       state["timestamp"],
        "terminal_status": status,
        "path_taken":      path,
        "policy_flags":    state.get("policy_flags", []),
        "hitl_action":     state.get("hitl_action"),
        "final_response":  final_resp,
        "booking_id":      "BK-XXXX",
    }
    try:
        FilesystemMiddleware.write_json(str(log_path), trace)
    except Exception as e:
        logger.error("[emit_outputs] Trace write failed: %s", e)

    return {**state, "final_response": final_resp, "path_taken": path}