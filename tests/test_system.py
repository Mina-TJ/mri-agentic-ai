"""
tests/test_system.py — Verification test suite for the MRI Reservation System.

Run with:
  python -m pytest tests/ -v
  python tests/test_system.py        (standalone)
"""

import sys
import os
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from state import MRIState
from middleware import PIIMiddleware, ModelRetryMiddleware, ToolCallLimitMiddleware
import nodes as n

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def make_state(**overrides) -> MRIState:
    base = MRIState(
        run_id="test-run-" + str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        user_message="Test message",
        intent="general",
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
    )
    base.update(overrides)
    return base


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────
# TEST 1: PIIMiddleware
# ─────────────────────────────────────────────
def test_pii_middleware():
    print("\n[1] PIIMiddleware")
    text = "Patient BK-1001 with DOB 1985-03-22 called from 902-555-0101."
    masked = PIIMiddleware.mask(text, "BK-1001")
    check("booking_id masked",   "BK-XXXX" in masked)
    check("DOB masked",          "****-**-**" in masked)
    check("phone masked",        "***-***-****" in masked)
    check("original BK gone",    "BK-1001" not in masked)


# ─────────────────────────────────────────────
# TEST 2: ToolCallLimitMiddleware
# ─────────────────────────────────────────────
def test_tool_limit():
    print("\n[2] ToolCallLimitMiddleware")
    mw = ToolCallLimitMiddleware(max_calls=3)
    run_id = "test-limit"
    try:
        for _ in range(3):
            mw.check(run_id, "test_tool")
        check("3 calls allowed", True)
    except RuntimeError:
        check("3 calls allowed", False)

    try:
        mw.check(run_id, "test_tool")   # 4th call — should raise
        check("4th call raises", False, "Expected RuntimeError")
    except RuntimeError:
        check("4th call raises", True)


# ─────────────────────────────────────────────
# TEST 3: ModelRetryMiddleware
# ─────────────────────────────────────────────
def test_retry_middleware():
    print("\n[3] ModelRetryMiddleware")
    mw = ModelRetryMiddleware(max_retries=3, base_delay=0.01)
    counter = {"n": 0}

    def flaky():
        counter["n"] += 1
        if counter["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    result = mw.call(flaky)
    check("retries and succeeds", result == "ok", f"after {counter['n']} attempts")

    def always_fail():
        raise ValueError("permanent")

    try:
        mw.call(always_fail)
        check("raises after max retries", False)
    except ValueError:
        check("raises after max retries", True)


# ─────────────────────────────────────────────
# TEST 4: risk_screen — urgent routing
# ─────────────────────────────────────────────
def test_risk_screen_urgent():
    print("\n[4] risk_screen — urgent routing")
    mock_client = MagicMock()
    mock_mod    = MagicMock()
    mock_mod.screen.return_value = {"flagged": False, "categories": {}, "logged": True}

    state = make_state(
        user_message="I have severe chest pain",
        is_urgent=True,
    )
    result = n.risk_screen(state, client=mock_client, moderation=mock_mod)
    check("terminal_status = ESCALATE",  result["terminal_status"] == "ESCALATE")
    check("risk_screen in path",         "risk_screen" in result["path_taken"])


# ─────────────────────────────────────────────
# TEST 5: identity_gate — prep_info bypass
# ─────────────────────────────────────────────
def test_identity_gate_prep_bypass():
    print("\n[5] identity_gate — prep_info bypass")
    state = make_state(intent="prep_info", booking_id=None)
    result = n.identity_gate(state)
    check("no terminal_status set",    result.get("terminal_status") == "")
    check("identity_gate in path",     "identity_gate" in result["path_taken"])
    check("booking_id still None",     result.get("booking_id") is None)


# ─────────────────────────────────────────────
# TEST 6: identity_gate — NEED_INFO on missing booking_id
# ─────────────────────────────────────────────
def test_identity_gate_need_info():
    print("\n[6] identity_gate — NEED_INFO on missing booking_id")
    state = make_state(intent="cancel", booking_id=None)
    result = n.identity_gate(state)
    check("terminal_status = NEED_INFO", result["terminal_status"] == "NEED_INFO")
    check("identity_gate in path",       "identity_gate" in result["path_taken"])


# ─────────────────────────────────────────────
# TEST 7: policy_gate — prep_info fast-path
# ─────────────────────────────────────────────
def test_policy_gate_prep_fastpath():
    print("\n[7] policy_gate — prep_info fast-path")
    state = make_state(intent="prep_info", hours_until_appt=10)
    result = n.policy_gate(state)
    check("hitl_reason is None",      result.get("hitl_reason") is None)
    check("policy_flags empty",       result.get("policy_flags") == [])
    check("policy_gate in path",      "policy_gate" in result["path_taken"])


# ─────────────────────────────────────────────
# TEST 8: policy_gate — late cancel HITL trigger
# ─────────────────────────────────────────────
def test_policy_gate_late_cancel():
    print("\n[8] policy_gate — late cancel triggers HITL")
    state = make_state(
        intent="cancel",
        hours_until_appt=5,
        patient_record={"appointment": {"reschedule_count": 0}},
    )
    result = n.policy_gate(state)
    check("late_cancel flag set",     "late_cancel" in result["policy_flags"])
    check("hitl_reason not None",     result.get("hitl_reason") is not None)


# ─────────────────────────────────────────────
# TEST 9: emit_outputs — NEED_INFO sets final_response
# ─────────────────────────────────────────────
def test_emit_outputs_need_info():
    print("\n[9] emit_outputs — NEED_INFO final_response")
    state = make_state(terminal_status="NEED_INFO", final_response="")
    result = n.emit_outputs(state)
    check("final_response not empty",  len(result["final_response"]) > 0)
    check("mentions booking ID",       "booking" in result["final_response"].lower())
    check("emit_outputs in path",      "emit_outputs" in result["path_taken"])


# ─────────────────────────────────────────────
# TEST 10: emit_outputs — ESCALATE sets final_response
# ─────────────────────────────────────────────
def test_emit_outputs_escalate():
    print("\n[10] emit_outputs — ESCALATE final_response")
    state = make_state(terminal_status="ESCALATE", final_response="")
    result = n.emit_outputs(state)
    check("final_response not empty",  len(result["final_response"]) > 0)
    check("mentions urgent/care",      "urgent" in result["final_response"].lower()
                                       or "immediate" in result["final_response"].lower())


# ─────────────────────────────────────────────
# TEST 11: escalate — deterministic stop
# ─────────────────────────────────────────────
def test_escalate_node():
    print("\n[11] escalate — deterministic stop, no downstream processing")
    state = make_state(
        terminal_status="ESCALATE",
        escalation_reason="Test escalation",
    )
    result = n.escalate(state)
    check("terminal_status preserved",  result["terminal_status"] == "ESCALATE")
    check("escalate in path",           "escalate" in result["path_taken"])
    check("draft_response untouched",   result.get("draft_response") == "")


# ─────────────────────────────────────────────
# TEST 12: Data file integrity
# ─────────────────────────────────────────────
def test_data_files():
    print("\n[12] Data file integrity")
    from middleware import FilesystemMiddleware

    data_dir = Path(__file__).parent.parent / "data"

    try:
        db = FilesystemMiddleware.read_json(str(data_dir / "patient_db.json"))
        check("patient_db.json readable",   True)
        check("patients key exists",        "patients" in db)
        check("at least 1 patient",         len(db["patients"]) >= 1)
    except Exception as e:
        check("patient_db.json readable",   False, str(e))

    try:
        prep = FilesystemMiddleware.read_json(str(data_dir / "prep_instructions.json"))
        check("prep_instructions.json readable", True)
        check("general prep exists",             "general" in prep.get("prep_instructions", {}))
    except Exception as e:
        check("prep_instructions.json readable", False, str(e))


# ─────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  MRI RESERVATION SYSTEM — TEST SUITE")
    print("═" * 60)

    test_pii_middleware()
    test_tool_limit()
    test_retry_middleware()
    test_risk_screen_urgent()
    test_identity_gate_prep_bypass()
    test_identity_gate_need_info()
    test_policy_gate_prep_fastpath()
    test_policy_gate_late_cancel()
    test_emit_outputs_need_info()
    test_emit_outputs_escalate()
    test_escalate_node()
    test_data_files()

    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"\n{'═'*60}")
    print(f"  RESULTS: {passed}/{total} passed")
    print("═" * 60 + "\n")
    sys.exit(0 if passed == total else 1)
