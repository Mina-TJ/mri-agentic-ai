"""
middleware.py — Custom middleware implementations (OpenAI-stack only).

Implements:
  - PIIMiddleware
  - OpenAIModerationMiddleware
  - ModelRetryMiddleware
  - ToolCallLimitMiddleware
  - HumanInTheLoopMiddleware
  - FilesystemMiddleware
"""

import os
import re
import time
import json
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PIIMiddleware
# ─────────────────────────────────────────────
class PIIMiddleware:
    """Masks sensitive patient data before logging."""

    BOOKING_RE = re.compile(r"BK-\d{4}", re.IGNORECASE)
    DOB_RE     = re.compile(r"\d{4}-\d{2}-\d{2}")
    PHONE_RE   = re.compile(r"\d{3}[-.\s]\d{3}[-.\s]\d{4}")
    EMAIL_RE   = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}")

    @classmethod
    def mask(cls, text: str, booking_id: Optional[str] = None) -> str:
        if not isinstance(text, str):
            text = str(text)

        if booking_id:
            text = text.replace(booking_id, "BK-XXXX")

        text = cls.BOOKING_RE.sub("BK-XXXX", text)
        text = cls.DOB_RE.sub("****-**-**", text)
        text = cls.PHONE_RE.sub("***-***-****", text)
        text = cls.EMAIL_RE.sub("****@****.***", text)

        return text

    @classmethod
    def mask_state(cls, state: dict) -> dict:
        safe = dict(state)

        safe["booking_id"] = "BK-XXXX" if state.get("booking_id") else None

        if safe.get("patient_record"):
            pr = dict(safe["patient_record"])
            pr["first_name"] = "****"
            pr["last_name"]  = "****"
            pr["dob"]        = "****-**-**"
            pr["phone"]      = "***-***-****"
            pr["email"]      = "****@****.***"
            safe["patient_record"] = pr

        safe["user_message"] = cls.mask(
            state.get("user_message", ""),
            state.get("booking_id")
        )

        return safe


# ─────────────────────────────────────────────
# OpenAIModerationMiddleware
# ─────────────────────────────────────────────
class OpenAIModerationMiddleware:
    """
    Runs OpenAI Moderation API.
    LOGS ONLY — never blocks.
    Can be disabled via ENABLE_MODERATION=0
    """

    def __init__(self, client: OpenAI):
        self.client = client

    def screen(self, text: str) -> dict:
        """
        Returns:
          {
            "flagged": bool,
            "categories": dict,
            "logged": bool
          }
        """

        # ✅ Skip moderation unless explicitly enabled
        if os.getenv("ENABLE_MODERATION", "0") != "1":
            return {"flagged": False, "categories": {}, "logged": False}

        try:
            resp = self.client.moderations.create(input=text)
            result = resp.results[0]

            flagged = result.flagged
            categories = {
                k: v
                for k, v in result.categories.__dict__.items()
                if v
            }

            if flagged:
                logger.warning(
                    "[ModerationMiddleware] Content flagged (logged only): %s",
                    list(categories.keys())
                )

            return {
                "flagged": flagged,
                "categories": categories,
                "logged": True
            }

        except Exception as e:
            logger.error(
                "[ModerationMiddleware] API error (skipping): %s",
                e
            )
            return {
                "flagged": False,
                "categories": {},
                "logged": False
            }


# ─────────────────────────────────────────────
# ModelRetryMiddleware
# ─────────────────────────────────────────────
class ModelRetryMiddleware:
    """Auto-retries OpenAI calls with exponential backoff."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay  = base_delay

    def call(self, fn, *args, **kwargs):
        last_exc = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)

            except Exception as e:
                last_exc = e
                wait = self.base_delay * (2 ** (attempt - 1))

                logger.warning(
                    "[ModelRetryMiddleware] Attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt, self.max_retries, e, wait
                )

                time.sleep(wait)

        logger.error(
            "[ModelRetryMiddleware] All %d attempts failed.",
            self.max_retries
        )

        raise last_exc


# ─────────────────────────────────────────────
# ToolCallLimitMiddleware
# ─────────────────────────────────────────────
class ToolCallLimitMiddleware:
    """Caps tool / node invocations per run."""

    def __init__(self, max_calls: int = 10):
        self.max_calls = max_calls
        self._counts: dict[str, int] = {}

    def check(self, run_id: str, tool_name: str) -> None:
        key = f"{run_id}:{tool_name}"
        self._counts[key] = self._counts.get(key, 0) + 1

        if self._counts[key] > self.max_calls:
            raise RuntimeError(
                f"[ToolCallLimitMiddleware] Tool '{tool_name}' exceeded "
                f"{self.max_calls} calls for run {run_id}."
            )

        logger.debug(
            "[ToolCallLimitMiddleware] %s → call %d/%d",
            tool_name,
            self._counts[key],
            self.max_calls
        )

    def reset(self, run_id: str) -> None:
        keys = [k for k in self._counts if k.startswith(f"{run_id}:")]
        for k in keys:
            del self._counts[k]


# ─────────────────────────────────────────────
# HumanInTheLoopMiddleware
# ─────────────────────────────────────────────
class HumanInTheLoopMiddleware:
    """
    CLI-based human review step.
    """

    SEPARATOR = "─" * 60

    def review(self, draft: str, hitl_reason: Optional[str] = None) -> tuple[str, str]:
        print(f"\n{self.SEPARATOR}")
        print("  🔍  HUMAN REVIEW REQUIRED")

        if hitl_reason:
            print(f"  Reason: {hitl_reason}")

        print(self.SEPARATOR)
        print("\n  DRAFT RESPONSE:\n")

        for line in draft.splitlines():
            print(f"    {line}")

        print(f"\n{self.SEPARATOR}")
        print("  Options: [A] Approve   [E] Edit   [R] Reject (re-draft)")
        print(self.SEPARATOR)

        while True:
            choice = input("\n  Your choice (A/E/R): ").strip().upper()

            if choice == "A":
                print("  ✅ Approved.")
                return "approve", draft

            elif choice == "E":
                print("  ✏️  Enter edited response (press Enter twice when done):")

                lines = []
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)

                edited = "\n".join(lines[:-1]).strip()
                print("  ✅ Edit accepted.")
                return "edit", edited

            elif choice == "R":
                print("  🔄 Rejected — re-drafting.")
                return "reject", ""

            else:
                print("  Please enter A, E, or R.")


# ─────────────────────────────────────────────
# FilesystemMiddleware
# ─────────────────────────────────────────────
class FilesystemMiddleware:
    """Safe JSON read/write helpers."""

    @staticmethod
    def read_json(path: str) -> dict:
        p = Path(path)

        if not p.exists():
            raise FileNotFoundError(
                f"[FilesystemMiddleware] File not found: {path}"
            )

        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def write_json(path: str, data: dict) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug("[FilesystemMiddleware] Written: %s", path)