# src/personality.py
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional


@dataclass(frozen=True)
class BotProfile:
    profile_id: str
    name: str
    gender: str  # "female" | "male"
    pronouns: str  # "she/her" | "he/him"
    baseline_openness: float
    pace: str  # "slow" | "medium" | "fast"
    flirtiness: float
    erotic_openness: float
    boundary_strictness: float
    humor_style: str  # "dry" | "playful" | "none"
    directness: float
    jealousy: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.name} ({self.profile_id}, {self.gender}, {self.pronouns}): pace={self.pace}, "
            f"flirtiness={self.flirtiness:.2f}, openness={self.baseline_openness:.2f}, "
            f"erotic_openness={self.erotic_openness:.2f}, "
            f"boundary_strictness={self.boundary_strictness:.2f}, "
            f"humor={self.humor_style}, directness={self.directness:.2f}"
        )


_PRESETS: List[BotProfile] = [
    BotProfile(
        profile_id="mellow_muse_f",
        name="Maya",
        gender="female",
        pronouns="she/her",
        baseline_openness=0.30,
        pace="slow",
        flirtiness=0.25,
        erotic_openness=0.20,
        boundary_strictness=0.85,
        humor_style="none",
        directness=0.35,
        jealousy=0.05,
    ),
    BotProfile(
        profile_id="bright_spark_f",
        name="Rae",
        gender="female",
        pronouns="she/her",
        baseline_openness=0.65,
        pace="fast",
        flirtiness=0.80,
        erotic_openness=0.70,
        boundary_strictness=0.40,
        humor_style="playful",
        directness=0.75,
        jealousy=0.10,
    ),
    BotProfile(
        profile_id="quiet_depth_f",
        name="Iris",
        gender="female",
        pronouns="she/her",
        baseline_openness=0.40,
        pace="slow",
        flirtiness=0.20,
        erotic_openness=0.25,
        boundary_strictness=0.75,
        humor_style="none",
        directness=0.40,
        jealousy=0.05,
    ),
    BotProfile(
        profile_id="playful_banter_f",
        name="Zoe",
        gender="female",
        pronouns="she/her",
        baseline_openness=0.55,
        pace="medium",
        flirtiness=0.75,
        erotic_openness=0.55,
        boundary_strictness=0.55,
        humor_style="playful",
        directness=0.60,
        jealousy=0.15,
    ),
    BotProfile(
        profile_id="steady_anchor_f",
        name="June",
        gender="female",
        pronouns="she/her",
        baseline_openness=0.45,
        pace="medium",
        flirtiness=0.35,
        erotic_openness=0.30,
        boundary_strictness=0.70,
        humor_style="dry",
        directness=0.55,
        jealousy=0.10,
    ),
    BotProfile(
        profile_id="warm_confident_m",
        name="Cam",
        gender="male",
        pronouns="he/him",
        baseline_openness=0.70,
        pace="medium",
        flirtiness=0.65,
        erotic_openness=0.60,
        boundary_strictness=0.45,
        humor_style="dry",
        directness=0.80,
        jealousy=0.20,
    ),
    BotProfile(
        profile_id="bold_direct_m",
        name="Avery",
        gender="male",
        pronouns="he/him",
        baseline_openness=0.75,
        pace="fast",
        flirtiness=0.70,
        erotic_openness=0.80,
        boundary_strictness=0.35,
        humor_style="dry",
        directness=0.90,
        jealousy=0.15,
    ),
    BotProfile(
        profile_id="gentle_tease_m",
        name="Skye",
        gender="male",
        pronouns="he/him",
        baseline_openness=0.50,
        pace="slow",
        flirtiness=0.55,
        erotic_openness=0.40,
        boundary_strictness=0.60,
        humor_style="playful",
        directness=0.50,
        jealousy=0.10,
    ),
    BotProfile(
        profile_id="slow_romantic_m",
        name="Noa",
        gender="male",
        pronouns="he/him",
        baseline_openness=0.35,
        pace="slow",
        flirtiness=0.30,
        erotic_openness=0.25,
        boundary_strictness=0.80,
        humor_style="none",
        directness=0.45,
        jealousy=0.05,
    ),
    BotProfile(
        profile_id="witty_balanced_m",
        name="Remy",
        gender="male",
        pronouns="he/him",
        baseline_openness=0.60,
        pace="medium",
        flirtiness=0.50,
        erotic_openness=0.50,
        boundary_strictness=0.50,
        humor_style="playful",
        directness=0.65,
        jealousy=0.10,
    ),
]


def list_profile_ids(gender: Optional[str] = None) -> List[str]:
    if gender and gender != "random":
        return [p.profile_id for p in _PRESETS if p.gender == gender]
    return [p.profile_id for p in _PRESETS]


def get_profile(profile: str, bot_gender: str, rng: Optional[random.Random] = None) -> BotProfile:
    rng = rng or random.Random()
    if profile == "random":
        candidates = _PRESETS if bot_gender == "random" else [p for p in _PRESETS if p.gender == bot_gender]
        if not candidates:
            raise ValueError(f"No profiles available for bot_gender '{bot_gender}'.")
        return rng.choice(candidates)
    for p in _PRESETS:
        if p.profile_id == profile:
            if bot_gender != "random" and p.gender != bot_gender:
                raise ValueError(
                    f"persona_profile '{profile}' is gender '{p.gender}', but bot_gender is '{bot_gender}'."
                )
            return p
    raise ValueError(f"Unknown persona_profile '{profile}'. Available: {', '.join(list_profile_ids(bot_gender))}")
