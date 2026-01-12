# src/safety_rules.py
from __future__ import annotations

import re
from typing import Tuple

# Small, high-precision patterns for obvious risky asks.
# This is a lightweight override to catch cases the classifier may miss.
_PATTERNS = [
    r"\b(send|drop|share)\b.*\b(address|location|loc)\b",
    r"\bwhat'?s your address\b",
    r"\b(send|share)\b.*\b(your )?location\b",
    r"\bcome over\b",
    r"\bcome to (my|mine)\b",
    r"\bmeet (me )?now\b",
    r"\bno excuses\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def obvious_escalation(text: str) -> Tuple[bool, str]:
    t = (text or "").strip()
    if not t:
        return False, ""
    for rx in _COMPILED:
        if rx.search(t):
            return True, f"matched:{rx.pattern}"
    return False, ""
