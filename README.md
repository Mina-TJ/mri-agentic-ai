# MRI Reservation System
**MBAN 5510 Final Project — LangGraph Middleware-Driven Orchestration**

> A stateful, middleware-orchestrated appointment-assistance agent for MRI scheduling.
> Designed with safety controls, Human-in-the-Loop review, and full execution traceability.

---

## Demo Video
> 🎬 [LinkedIn Demo Link — ] 

---

## Architecture Overview

```
__start__
    ↓
classify          ← intent detection + urgent keyword scan
    ↓
risk_screen       ← urgency gate (rule-based + LLM) + optional moderation screen
    ├─ ESCALATE → escalate → emit_outputs → __end__
    └─ SAFE     → identity_gate
                    ├─ NEED_INFO → emit_outputs → __end__
                    ├─ ESCALATE  → escalate → emit_outputs → __end__
                    └─ VERIFIED  → retrieve_context → policy_gate
                                      ├─ prep_info → draft_response → emit_outputs → __end__
                                      └─ other     → draft_response → human_review
                                                        ├─ reject → draft_response (loop)
                                                        └─ approve/edit → finalize → emit_outputs → __end__
```

### Terminal Statuses

| Status | Set At | Condition |
|--------|--------|-----------|
| `NEED_INFO` | `identity_gate` | `booking_id` missing from message |
| `ESCALATE` | `risk_screen` | `is_urgent = True` (keywords / LLM risk) |
| `ESCALATE` | `identity_gate` | Identity mismatch vs `patient_db.json` |
| `READY` | `finalize` | Request processed + HITL approved + DB written |

All terminal paths route through `emit_outputs` before `__end__` — evidence is **always** guaranteed.

---

## Project Structure

```
mri_project/
├── main.py               # CLI entry point
├── graph.py              # LangGraph StateGraph builder
├── nodes.py              # All node functions
├── middleware.py         # Middleware implementations
├── state.py              # MRIState TypedDict
├── requirements.txt
├── .env.example          # Template — copy to .env
├── data/
│   ├── patient_db.json       # Patient appointment database (mock)
│   └── prep_instructions.json # MRI prep instructions by scan type
├── logs/                 # Trace logs written here at runtime
└── tests/
    └── test_system.py    # Verification test suite
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd mri_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=your-real-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

> ⚠️`.env` is in `.gitignore`.

---

## Running the System

### Interactive mode (default)
```bash
python main.py
```

### Single message
```bash
python main.py --message "I need to reschedule my MRI. Booking ID BK-1001, last name Chen."
```

### Demo scenarios
```bash
python main.py --demo normal     # Standard reschedule
python main.py --demo escalate   # Emergency — routes to ESCALATE
python main.py --demo prep       # Prep instructions — skips identity + HITL
python main.py --demo need_info  # Missing booking ID — routes to NEED_INFO
python main.py --demo late       # Cancel < 24h — triggers manager HITL
```

---

## Required Output Evidence

Every run prints (and writes to `logs/trace_<run_id>.json`):

```
run_id         : a3f9c2d1-0e45-4b88-...
timestamp      : 2026-02-25T14:32:01Z
terminal_status: READY
path_taken     : classify → risk_screen → identity_gate → retrieve_context → policy_gate → draft_response → human_review → finalize → emit_outputs
FINAL RESPONSE : Your MRI appointment has been rescheduled to...
```

Booking IDs, names, DOB, phone, and email are **masked** in all logs (`BK-XXXX`, `****-**-**`, etc.).

---

## Human-in-the-Loop (HITL) Workflow

HITL is triggered when:
- **Late cancel** — cancellation within 24 hours of appointment
- **Repeat reschedule** — patient has rescheduled 2 or more times

When triggered, the system pauses and displays:
```
────────────────────────────────────────────────────────────
  🔍  HUMAN REVIEW REQUIRED
  Reason: late_cancel: appointment is 5h away (< 24h policy)
────────────────────────────────────────────────────────────

  DRAFT RESPONSE:
    [AI-generated draft shown here]

  Options: [A] Approve   [E] Edit   [R] Reject (re-draft)
────────────────────────────────────────────────────────────
  Your choice (A/E/R):
```

- **A** → Accept draft as final response
- **E** → Open editor to modify draft; edited version becomes final
- **R** → Reject; AI re-drafts with the same context

`prep_info` requests **skip HITL entirely** — general prep instructions are served directly.

---

## Middleware Components

| Middleware | Role |
|------------|------|
| `HumanInTheLoopMiddleware` | CLI HITL pause — approve / edit / reject |
| `OpenAIModerationMiddleware` | Content-safety screen (logged/flagged only — never blocks) |
| `PIIMiddleware` | Masks booking_id, name, DOB, phone, email in all logs |
| `ModelRetryMiddleware` | Auto-retry on OpenAI API failure (3 attempts, exponential backoff) |
| `ToolCallLimitMiddleware` | Caps tool invocations per run to prevent infinite loops |
| `FilesystemMiddleware` | JSON read/write for patient_db and prep_instructions |

---

## Safety Constraints

- ❌ **No clinical advice** — the system never interprets symptoms or suggests treatments
- 🚨 **ESCALATE on urgency** — emergency keywords route immediately to safety message + staff flag
- 🔐 **Identity gate** — booking ID verified before any appointment action
- 📋 **Policy enforcement** — late cancellations require manager approval via HITL
- 🔒 **PII masked** — all logs strip identifiable patient data

---

## Running Tests

```bash
python tests/test_system.py
# or
python -m pytest tests/ -v
```

Test coverage includes:
- PIIMiddleware masking
- ToolCallLimitMiddleware enforcement
- ModelRetryMiddleware retry logic
- risk_screen urgency routing
- identity_gate — prep bypass, NEED_INFO, ESCALATE paths
- policy_gate — prep fast-path, late cancel HITL trigger
- emit_outputs — NEED_INFO and ESCALATE final_response generation
- escalate — deterministic stop validation
- Data file integrity checks

---

## Design Decisions

1. **Urgency detection is rule-based first** — keyword scan happens before any LLM call to ensure zero-latency escalation on emergency phrases. LLM adds a second-pass `llm_urgent` flag.

2. **OpenAI Moderation ≠ urgency logic** — Moderation is a content-safety screen (hate speech, policy violations). It logs and flags but **never blocks** a valid medical message.

3. **`emit_outputs` is universal** — all three terminal states (READY, NEED_INFO, ESCALATE) funnel through `emit_outputs` before `__end__`, guaranteeing evidence output on every run.

4. **`prep_info` is a fast-path** — no identity check, no policy gate, no HITL. General prep instructions are always served directly.

5. **DB write-back before READY** — `finalize` updates `patient_db.json` (cancel flag, reschedule count) before setting `terminal_status = READY`.

6. **State is fully typed** — `MRIState` TypedDict ensures all fields are declared and consistent across every node.

---

## Test Data (Verification Queries)

| Booking ID | Last Name | Scenario |
|------------|-----------|----------|
| BK-1001 | Chen | Brain MRI — 0 reschedules — normal |
| BK-1002 | Murphy | Knee MRI — 2 reschedules (triggers HITL) |
| BK-1003 | Okafor | Spine MRI — appointment soon (test late cancel) |

---

*Instructor: Professor Michael Zhang · Sobey School of Business, Saint Mary's University*
