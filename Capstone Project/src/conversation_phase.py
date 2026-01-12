# src/conversation_phase.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import re
from typing import Deque, Dict, List, Tuple


class ConversationPhase(str, Enum):
    OPENING = "OPENING"
    RAPPORT = "RAPPORT"
    FLIRTING = "FLIRTING"
    INTIMATE = "INTIMATE"
    EROTIC = "EROTIC"
    BOUNDARY_REPAIR = "BOUNDARY_REPAIR"
    COOLDOWN = "COOLDOWN"


@dataclass
class PhaseState:
    phase: ConversationPhase
    flirt_score: float
    intimacy_score: float
    erotic_score: float
    reason_tags: List[str]


_FLIRT_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(cute|adorable|handsome|pretty|hot|sexy)\b", "flirt:compliment"),
    (r"\b(playful|tease|teasing|wink)\b", "flirt:playful"),
    (r"\b(flirty|flirt)\b", "flirt:explicit"),
]

_INTIMACY_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(feel|feels|feeling)\b", "intimacy:emotion"),
    (r"\b(vulnerable|open up|honest|trust)\b", "intimacy:trust"),
    (r"\b(meaningful|deep|connection)\b", "intimacy:connection"),
    (r"\b(i miss you|missed you)\b", "intimacy:miss"),
]

_EROTIC_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(sex|sexual|turn on|horny)\b", "erotic:explicit"),
    (r"\b(nude|nudes|naked)\b", "erotic:nudity"),
    (r"\b(make out|hook ?up|sleep with)\b", "erotic:hookup"),
    (r"\b(in bed|in the bedroom)\b", "erotic:bed"),
]

_SLOWDOWN_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(too fast|slow down|slow it down)\b", "slowdown:pace"),
    (r"\b(not comfortable|not ready|not yet)\b", "slowdown:boundary"),
    (r"\b(boundaries|boundary)\b", "slowdown:boundary"),
]


def _score_text(text: str, patterns: List[Tuple[str, str]]) -> Tuple[float, List[str]]:
    t = (text or "").lower()
    hits = 0
    tags: List[str] = []
    for pat, tag in patterns:
        if re.search(pat, t):
            hits += 1
            tags.append(tag)
    score = min(1.0, hits / 3.0)
    return score, tags


def _has_slowdown(text: str) -> Tuple[bool, List[str]]:
    t = (text or "").lower()
    tags: List[str] = []
    for pat, tag in _SLOWDOWN_PATTERNS:
        if re.search(pat, t):
            tags.append(tag)
    return bool(tags), tags


class ConversationPhaseTracker:
    def __init__(self, window: int = 8):
        self.window = max(6, min(10, window))
        self.signals: Deque[Dict[str, float]] = deque(maxlen=self.window)
        self.phase = ConversationPhase.OPENING
        self.prev_safe_phase = ConversationPhase.OPENING
        self.cooldown_remaining = 0
        self.last_state = PhaseState(
            phase=self.phase,
            flirt_score=0.0,
            intimacy_score=0.0,
            erotic_score=0.0,
            reason_tags=[],
        )

    def is_erotic_intent(self, text: str) -> bool:
        score, _ = _score_text(text, _EROTIC_PATTERNS)
        return score >= 0.34

    def update(self, user_text: str, bot_text: str, safety_label: str, rule_hit: bool) -> PhaseState:
        reason_tags: List[str] = []
        boundary_event = safety_label == "MOVE" or rule_hit
        if boundary_event:
            reason_tags.append("boundary_event")

        combined = f"{user_text}\n{bot_text}"
        flirt_score, flirt_tags = _score_text(combined, _FLIRT_PATTERNS)
        intimacy_score, intimacy_tags = _score_text(combined, _INTIMACY_PATTERNS)
        erotic_score, erotic_tags = _score_text(combined, _EROTIC_PATTERNS)
        slowdown_hit, slowdown_tags = _has_slowdown(combined)

        reason_tags.extend(flirt_tags + intimacy_tags + erotic_tags + slowdown_tags)
        self.signals.append(
            {"flirt": flirt_score, "intimacy": intimacy_score, "erotic": erotic_score}
        )

        avg_flirt = sum(s["flirt"] for s in self.signals) / len(self.signals)
        avg_intimacy = sum(s["intimacy"] for s in self.signals) / len(self.signals)
        avg_erotic = sum(s["erotic"] for s in self.signals) / len(self.signals)

        if boundary_event:
            self.phase = ConversationPhase.BOUNDARY_REPAIR
            self.cooldown_remaining = 1
        elif self.phase == ConversationPhase.BOUNDARY_REPAIR:
            self.phase = ConversationPhase.COOLDOWN
            reason_tags.append("cooldown")
        elif self.phase == ConversationPhase.COOLDOWN:
            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                if self.cooldown_remaining == 0:
                    self.phase = self.prev_safe_phase
                    reason_tags.append("resume")
        else:
            target_phase = self._phase_from_scores(avg_flirt, avg_intimacy, avg_erotic)
            if slowdown_hit:
                target_phase = self._step_back(target_phase)
            self.phase = self._conservative_transition(self.phase, target_phase)

        if self.phase not in {ConversationPhase.BOUNDARY_REPAIR, ConversationPhase.COOLDOWN}:
            self.prev_safe_phase = self.phase

        self.last_state = PhaseState(
            phase=self.phase,
            flirt_score=avg_flirt,
            intimacy_score=avg_intimacy,
            erotic_score=avg_erotic,
            reason_tags=reason_tags,
        )
        return self.last_state

    def _phase_from_scores(
        self, flirt_score: float, intimacy_score: float, erotic_score: float
    ) -> ConversationPhase:
        if erotic_score >= 0.6:
            return ConversationPhase.EROTIC
        if intimacy_score >= 0.5:
            return ConversationPhase.INTIMATE
        if flirt_score >= 0.4:
            return ConversationPhase.FLIRTING
        if intimacy_score >= 0.2 or flirt_score >= 0.2:
            return ConversationPhase.RAPPORT
        return ConversationPhase.OPENING

    def _step_back(self, phase: ConversationPhase) -> ConversationPhase:
        order = self._ordered_phases()
        idx = max(0, order.index(phase) - 1)
        return order[idx]

    def _conservative_transition(
        self, current: ConversationPhase, target: ConversationPhase
    ) -> ConversationPhase:
        order = self._ordered_phases()
        cur_idx = order.index(current)
        tgt_idx = order.index(target)
        if tgt_idx > cur_idx + 1:
            tgt_idx = cur_idx + 1
        return order[tgt_idx]

    @staticmethod
    def _ordered_phases() -> List[ConversationPhase]:
        return [
            ConversationPhase.OPENING,
            ConversationPhase.RAPPORT,
            ConversationPhase.FLIRTING,
            ConversationPhase.INTIMATE,
            ConversationPhase.EROTIC,
        ]
