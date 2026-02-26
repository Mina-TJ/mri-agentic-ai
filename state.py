# state.py — MRIState TypedDict definition
from typing import Optional, List
from typing_extensions import TypedDict


class MRIState(TypedDict):
    # ── Entry ─────────────────────────────────────────
    run_id:            str
    timestamp:         str
    user_message:      str

    # ── Conversation / UX ─────────────────────────────
    tone:              str            # "friendly" | "neutral"
    need_fields:       List[str]      # e.g. ["booking_id", "last_name", "dob", "scan_type"]

    # ── Patient-provided fields (interactive) ─────────
    first_name_input:  Optional[str]
    last_name_input:   Optional[str]
    dob_input:         Optional[str]  # YYYY-MM-DD
    scan_type_input:   Optional[str]  # e.g. brain_mri, knee_mri, spine_mri, or "general"

    # ── Classify ──────────────────────────────────────
    intent:            str
    booking_id:        Optional[str]
    is_urgent:         bool

    # ── Identity ──────────────────────────────────────
    identity_verified: bool

    # ── Context ───────────────────────────────────────
    patient_record:    Optional[dict]
    hours_until_appt:  Optional[int]
    context_notes:     str

    # ── Policy ────────────────────────────────────────
    policy_flags:      List[str]
    hitl_reason:       Optional[str]

    # ── Draft / HITL ──────────────────────────────────
    draft_response:    str
    hitl_action:       str
    final_response:    str

    # ── Output ────────────────────────────────────────
    terminal_status:   str
    path_taken:        List[str]
    escalation_reason: Optional[str]

    # Internal
    _model:            str