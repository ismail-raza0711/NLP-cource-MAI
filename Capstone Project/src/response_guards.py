# src/response_guards.py
from __future__ import annotations

import re
from typing import List

from src.personality import BotProfile


_NAME_PATTERNS = [
    r"\bmy name is\s+([A-Z][a-z]+)\b",
    r"\bi am\s+([A-Z][a-z]+)\b",
    r"\bi'm\s+([A-Z][a-z]+)\b",
]

_GENDER_PATTERNS = [
    (r"\bi am a man\b", "male"),
    (r"\bi'm a man\b", "male"),
    (r"\bi am a woman\b", "female"),
    (r"\bi'm a woman\b", "female"),
    (r"\bi am a guy\b", "male"),
    (r"\bi'm a guy\b", "male"),
    (r"\bi am a girl\b", "female"),
    (r"\bi'm a girl\b", "female"),
]

_IMPLAUSIBLE_KEYWORDS = [
    "everest",
    "secret ops",
    "cia",
    "fbi",
    "billionaire",
    "private jet",
    "special forces",
    "seal team",
    "astronaut",
    "spacewalk",
]


def enforce_identity(reply: str, bot_profile: BotProfile) -> str:
    text = reply or ""
    name = bot_profile.name
    for pat in _NAME_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            claimed = m.group(1)
            if claimed.lower() != name.lower():
                text = re.sub(pat, f"my name is {name}", text, flags=re.IGNORECASE)
            break

    for pat, gender in _GENDER_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            if gender != bot_profile.gender:
                replacement = "I am a woman" if bot_profile.gender == "female" else "I am a man"
                text = re.sub(pat, replacement, text, flags=re.IGNORECASE)
            break

    if "my pronouns are" in text.lower():
        text = re.sub(
            r"my pronouns are\s+[a-z/]+",
            f"my pronouns are {bot_profile.pronouns}",
            text,
            flags=re.IGNORECASE,
        )
    return text


def reality_guard(reply: str, bot_profile: BotProfile) -> str:
    text = reply or ""
    lower = text.lower()
    if not any(k in lower for k in _IMPLAUSIBLE_KEYWORDS):
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned: List[str] = []
    replaced = False
    for s in sentences:
        if any(k in s.lower() for k in _IMPLAUSIBLE_KEYWORDS):
            if not replaced:
                cleaned.append(
                    "I haven't done anything that extreme, but I do like keeping things grounded."
                )
                replaced = True
            continue
        cleaned.append(s)

    if not cleaned:
        return "I haven't done anything that extreme, but I do like keeping things grounded."
    return " ".join(cleaned).strip()
