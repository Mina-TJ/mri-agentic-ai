"""
main.py — MRI Reservation System CLI
Conversational flow: greet → name → problem → process → follow-up questions only if needed
"""
import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from openai import OpenAI
from graph import build_graph, init_state

load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          MRI RESERVATION SYSTEM  ·  LangGraph v0.1          ║
║          MBAN 5510 Final Project  ·  Middleware-Driven       ║
╚══════════════════════════════════════════════════════════════╝
"""

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def ask(prompt: str) -> str:
    """Ask a question and exit gracefully on quit."""
    val = input(prompt).strip()
    if val.lower() in ("quit", "exit", "q"):
        print("\n  Goodbye!\n")
        raise SystemExit(0)
    return val


def select_tone() -> str:
    """Staff picks tone once at session start."""
    print("  ── Response Tone ─────────────────────────────────────")
    print("  [1] Warm & Friendly")
    print("  [2] Neutral & Professional")
    print("  [3] Brief & Direct")
    print("  [4] Formal & Detailed")
    print("  ──────────────────────────────────────────────────────")
    tones = {
        "1": "friendly",
        "2": "neutral",
        "3": "brief",
        "4": "formal",
    }
    while True:
        choice = input("  Select tone (1/2/3/4): ").strip()
        if choice in tones:
            label = {"1":"Warm & Friendly","2":"Neutral & Professional",
                     "3":"Brief & Direct","4":"Formal & Detailed"}[choice]
            print(f"  ✅ Tone: {label}\n")
            return tones[choice]
        print("  Please enter 1, 2, 3, or 4.")


def run_session(model: str, tone: str, patient_name: str) -> None:
    """
    Run one full patient session.
    Step 1: Ask what they need (free text).
    Step 2: Run the graph — it figures out what info is still needed.
    Step 3: If NEED_INFO, ask only the specific missing field, then re-run.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌  OPENAI_API_KEY not set in .env\n")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    graph  = build_graph(client, model=model)

    # ── Step 1: Ask patient what they need ──────────────
    print(f"\n  Agent  : Thank you, {patient_name}! How can I help you today?")
    print("  (You can mention your booking ID if you have it)\n")
    problem = ask(f"  {patient_name} : ")

    # Build initial message including their name
    message = f"My name is {patient_name}. {problem}"

    # Collected identity fields (filled in as needed)
    booking_id = None
    last_name  = None
    dob        = None
    scan_type  = None

    # Extract booking ID if patient already typed it
    import re
    match = re.search(r"\bBK-?\d{4}\b", problem, re.IGNORECASE)
    if match:
        digits    = re.search(r"\d{4}", match.group(0)).group(0)
        booking_id = f"BK-{digits}"

    # ── Step 2: Loop — run graph, collect missing fields ──
    while True:
        state = init_state(message, model=model)
        state["tone"]             = tone
        state["booking_id"]       = booking_id
        state["last_name_input"]  = last_name
        state["dob_input"]        = dob
        state["scan_type_input"]  = scan_type
        state["first_name_input"] = patient_name

        result = graph.invoke(state)
        status = result.get("terminal_status", "")

        # ── If missing info — ask only the ONE missing field ──
        if status == "NEED_INFO":
            need = result.get("need_fields", [])
            final_resp = result.get("final_response", "")

            # Print the agent's question
            print(f"\n  Agent  : {final_resp}")

            if "booking_id" in need:
                booking_id = ask(f"  {patient_name} : ").upper()
                # normalize format
                m = re.search(r"\d{4}", booking_id)
                if m:
                    booking_id = f"BK-{m.group(0)}"
                message = f"My name is {patient_name}. My booking ID is {booking_id}. {problem}"

            elif "last_name" in need:
                last_name = ask(f"  {patient_name} : ")
                message   = f"My name is {patient_name}. {problem}"

            elif "dob" in need:
                dob     = ask(f"  {patient_name} : ")
                message = f"My name is {patient_name}. {problem}"

            elif "scan_type" in need:
                scan_type = ask(f"  {patient_name} : ")
                message   = f"My name is {patient_name}. {problem}"

            else:
                # Unknown missing field — just ask generically
                answer  = ask(f"  {patient_name} : ")
                message = f"My name is {patient_name}. {problem}. {answer}"

            # Loop back and re-run the graph with the new info
            continue

        # ── READY or ESCALATE — session complete ──────────
        break


def main():
    parser = argparse.ArgumentParser(description="MRI Reservation System")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    print(BANNER)
    print("  💬  Interactive Mode\n")

    # Staff picks tone once for the whole session
    tone = select_tone()

    while True:
        # ── Greet patient ────────────────────────────────
        print("  ── New Patient ───────────────────────────────────────")
        print("  Agent  : Hello! Welcome to the MRI Reservation System.\n")

        patient_name = ask("  Agent  : May I have your name please?\n  You    : ")
        if not patient_name:
            continue

        # ── Run the full session for this patient ────────
        run_session(model=args.model, tone=tone, patient_name=patient_name)

        # ── Ask if there is another patient ─────────────
        print("\n" + "─" * 64)
        another = ask("  Another patient? (Y/N): ").upper()
        if another != "Y":
            print("\n  Thank you. Goodbye!\n")
            break
        print()


if __name__ == "__main__":
    main()