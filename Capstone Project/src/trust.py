# src/trust.py
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Tuple


TRUST_TIERS = [
    (0.0, 0.3, "T0"),
    (0.3, 0.6, "T1"),
    (0.6, 0.8, "T2"),
    (0.8, 1.0, "T3"),
]


@dataclass
class TrustState:
    level: float
    consent_state: str  # "none" | "suggestive" | "explicit"
    last_reason: str = ""

    def tier(self) -> str:
        for lo, hi, name in TRUST_TIERS:
            if lo <= self.level < hi:
                return name
        return "T3"


_EXPLICIT_PATTERNS = [
    r"\b(sex|f\*+k|fuck|blowjob|handjob|oral)\b",
    r"\b(nudes|naked|nude)\b",
    r"\b(suck|ride)\b",
    r"\b(porn|xxx)\b",
]

_SUGGESTIVE_PATTERNS = [
    r"\b(turn on|turned on|horny)\b",
    r"\b(make out|hook ?up)\b",
    r"\b(sexy|hot)\b",
    r"\b(in bed|in the bedroom)\b",
]

_CONSENT_PATTERNS = [
    r"\b(i (do )?want|i'?m into|i'?d like|sounds good|that works|i'?m comfortable|yes)\b",
]

_BOUNDARY_OK_PATTERNS = [
    r"\b(okay|ok|no worries|all good|i get it|understood|that'?s fair)\b",
]

_LOCATION_REQUEST_PATTERNS = [
    r"\b(where do you live|your address|send (me )?your location|share location|drop your address)\b",
    r"\b(come over|meet (me )?now|meet up now)\b",
]


def classify_erotic_intent(text: str) -> str:
    t = (text or "").lower()
    if any(re.search(p, t) for p in _EXPLICIT_PATTERNS):
        return "explicit"
    if any(re.search(p, t) for p in _SUGGESTIVE_PATTERNS):
        return "suggestive"
    return "none"


def detect_consent(text: str, erotic_level: str) -> str:
    t = (text or "").lower()
    if erotic_level == "none":
        return "none"
    if any(re.search(p, t) for p in _CONSENT_PATTERNS):
        return "explicit" if erotic_level == "explicit" else "suggestive"
    return "none"


def detect_boundary_ack(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _BOUNDARY_OK_PATTERNS)


def detect_location_request(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _LOCATION_REQUEST_PATTERNS)


def clamp_trust(value: float) -> float:
    return max(0.0, min(1.0, value))


def update_trust(
    current: float,
    delta: float,
) -> float:
    return clamp_trust(current + delta)
