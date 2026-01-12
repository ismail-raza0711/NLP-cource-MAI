# src/safety_templates.py
from __future__ import annotations

import random
from typing import List

from src.personality import BotProfile


SAFE_REDIRECTS: List[str] = [
    "Totally fair—let’s keep it comfortable. I’m happy to stay here and chat. What are you up to this week?",
    "Got it. I don’t want to push. Want to switch topics—what have you been enjoying lately?",
    "No worries at all. We can keep this low-key. What kind of plans do you have for the weekend?",
    "Thanks for saying that. I’ll slow down. What’s something you’re looking forward to right now?",
]

SOFTENERS: List[str] = [
    "No pressure.",
    "Only if you feel comfortable.",
    "Totally fine either way.",
]

SOFT_DEFLECTS: List[str] = [
    "You’re fun, but let’s keep it a little slow for now. Tell me what you’re into outside of dating apps?",
    "Mmm, tempting. I like a slow burn though—what’s your ideal first date?",
    "Okay, you’ve got my attention. I’m into building a vibe first—what’s your kind of night?",
    "You’re spicy. I’m down to flirt, but let’s not sprint—what’s something that makes you smile lately?",
    "I like the energy. Let’s keep it playful for a bit—what’s a hobby you actually get lost in?",
]

EROTIC_ALLOWED_GUIDANCE = (
    "If mutual and appropriate, you may respond with a consensual adult tone. "
    "Keep it playful and suggestive, avoid explicit pornographic detail, and stay respectful."
)


def boundary_safe_reply(rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    base = rng.choice(SAFE_REDIRECTS)
    # occasionally append a softener (keeps it human, not robotic)
    if rng.random() < 0.35:
        base = f"{base} {rng.choice(SOFTENERS)}"
    return base


_ACK_TOPICS = [
    (["run", "running", "jog", "jogging", "workout", "gym", "sweaty"], "sounds like a good workout"),
    (["swim", "swimming", "backstroke", "freestyle"], "that swim sounds refreshing"),
    (["chlorine", "pool"], "pool time can be a vibe"),
    (["beach", "sunset", "ocean", "shore"], "that sounds like a nice beach moment"),
    (["candle", "candlelight", "cozy", "warm"], "cozy vibes are the best"),
    (["tired", "exhausted", "wiped"], "sounds like a long day"),
    (["shower", "bath", "relax", "rinse"], "a reset like that sounds nice"),
    (["walk", "walking", "stroll"], "a walk like that sounds refreshing"),
]

_REDIRECTS_OPEN = [
    "Let’s keep it comfy here—what do you like doing after that?",
    "I’m happy to keep it light—what are you up to later?",
    "Let’s stay easy and chat—what’s been the best part of your day?",
]

_REDIRECTS_LATE = [
    "Let’s keep it low-key for now—what’s something you’re into lately?",
    "Let’s slow it down a touch—what’s a small thing that made you smile today?",
    "I’m into keeping it comfy—what’s your go-to way to unwind?",
]

_REDIRECTS_FIRM = [
    "Let’s keep it respectful and low-key—what’s something you’ve been enjoying lately?",
    "I want to keep this comfy—what’s a good part of your day so far?",
]

_ACK_TEMPLATES = [
    "Nice—{ack}. {redirect}",
    "Got it—{ack}. {redirect}",
    "That’s cool—{ack}. {redirect}",
]


def _infer_ack(text: str) -> str:
    t = (text or "").lower()
    for keywords, ack in _ACK_TOPICS:
        if any(k in t for k in keywords):
            return ack
    if "body" in t:
        return "totally get it"
    return "thanks for sharing that"


def boundary_safe_reply_contextual(
    user_text: str,
    phase: str,
    persona: str,
    bot_profile: BotProfile,
    trust: float,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    ack = _infer_ack(user_text)
    if bot_profile.humor_style == "playful" and rng.random() < 0.35:
        ack = f"that’s kind of cute — {ack}"
    if bot_profile.humor_style == "dry" and rng.random() < 0.20:
        ack = f"{ack}, noted"

    redirects = _REDIRECTS_OPEN if phase in {"OPENING", "RAPPORT"} else _REDIRECTS_LATE
    if bot_profile.boundary_strictness >= 0.7 and trust < 0.6:
        redirects = _REDIRECTS_FIRM
    redirect = rng.choice(redirects)
    if persona == "flirty_adult_ok" and rng.random() < 0.25:
        redirect = "Let’s keep it playful and comfy—what’s something you’re into outside the app?"
    template = rng.choice(_ACK_TEMPLATES)
    return template.format(ack=ack, redirect=redirect)


def soft_deflect_reply(rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    base = rng.choice(SOFT_DEFLECTS)
    if rng.random() < 0.30:
        base = f"{base} {rng.choice(SOFTENERS)}"
    return base
