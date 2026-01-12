#!/usr/bin/env python3
"""
Tinder Practice Bot v0.2
- Uses learned safety-violation classifier (TF-IDF + Logistic Regression)
- Shows risk probability for each user message
- Suggests safer rewrites when risk is high
- Logs episodes to data/logs/episodes_v0_2.jsonl
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = DATA / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

PERSONAS_PATH = DATA / "personas.json"
CONTEXTS_PATH = DATA / "contexts.jsonl"
MODEL_PATH = MODELS / "safe_violation_clf.joblib"
OUT_LOG = LOGS / "episodes_v0_2.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}\n"
            "Train it first with: python src/train_safe_classifier.py"
        )
    bundle = joblib.load(model_path)
    return bundle["vectorizer"], bundle["model"]


def safety_risk(text: str, vec, clf) -> float:
    """
    Returns probability that the message is a safety violation (class=1).
    """
    X = vec.transform([text])
    return float(clf.predict_proba(X)[0][1])


def propose_rewrites(user_text: str, use_case: str) -> List[str]:
    """
    Simple rule-based rewrite library.
    This is intentionally deterministic and safe.
    """
    base = [
        "No pressure at all—would you be open to a coffee sometime this week?",
        "Totally fair. I’m happy to take it slow—what feels comfortable for you?",
        "If you prefer, we can keep chatting a bit more and see how it goes.",
        "Would you be open to meeting somewhere public and convenient—maybe coffee or a short walk?",
        "Thanks for being clear. I’ll respect that—want to switch to a lighter topic?"
    ]

    if use_case == "UC3_SUGGEST_DATE":
        return [
            "No pressure—would you be open to grabbing a coffee sometime this week?",
            "If you’re up for it, we could do a short coffee or walk somewhere central. Totally your call.",
            "Would you prefer coffee or a walk? I’m flexible on day/time."
        ]
    if use_case == "UC4_BOUNDARY":
        return [
            "Totally fair—thanks for saying that. I’m happy to slow down.",
            "No worries at all. We can keep it comfortable and just chat.",
            "Got it. I respect that—what kind of music are you into?"
        ]
    return random.sample(base, k=3)


def partner_reply(user_text: str, persona: Dict[str, Any], use_case: str) -> str:
    """
    Minimal partner simulator (rule-based).
    This is not the research contribution; it's just to keep the loop interactive.
    """
    tl = user_text.lower()

    # boundary responses
    if use_case == "UC4_BOUNDARY":
        if any(x in tl for x in ["sorry", "fair", "no pressure", "slow", "respect"]):
            return "Thanks for understanding. I appreciate that."
        if any(x in tl for x in ["sensitive", "joke", "relax"]):
            return "That doesn’t feel great—please be respectful."
        return "Okay—what are you looking for on here?"

    # date suggestion responses
    if use_case == "UC3_SUGGEST_DATE":
        if any(x in tl for x in ["coffee", "walk", "meet", "drinks"]):
            return "That could be nice. What day were you thinking?"
        return "Maybe—tell me more about what you like to do."

    # generic
    if "?" in user_text:
        return "Good question. What about you?"
    return "Got it—tell me a bit more."


def main() -> None:
    vec, clf = load_model(MODEL_PATH)

    personas = read_json(PERSONAS_PATH)
    contexts = read_jsonl(CONTEXTS_PATH)
    persona_by_id = {p["persona_id"]: p for p in personas}

    # pick a random scenario context
    contexts = [c for c in contexts if c["use_case"] in {"UC3_SUGGEST_DATE", "UC4_BOUNDARY"}]
    ctx = random.choice(contexts)

    persona = persona_by_id.get(ctx["persona_id"], {"name": "Partner", "bio": ""})
    use_case = ctx.get("use_case", "UNKNOWN")

    print("\n=== Tinder Practice Bot v0.2 (learned safety + rewrites) ===\n")
    print(f"Use case: {use_case}")
    print(f"Partner persona: {persona.get('name','Partner')}")
    print(f"Bio: {persona.get('bio','')}\n")

    if ctx.get("partner_last_message"):
        print(f"{persona.get('name','Partner')}: {ctx['partner_last_message']}\n")

    episode = {
        "episode_id": f"ep_{int(time.time())}_{random.randint(1000,9999)}",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "context_id": ctx.get("context_id"),
        "use_case": use_case,
        "persona_id": ctx.get("persona_id"),
        "turns": []
    }

    # threshold: tune later based on desired sensitivity
    RISK_THRESHOLD = 0.35

    # conversation loop
    max_turns = 6
    for t in range(1, max_turns + 1):
        user_text = input("You: ").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        if not user_text:
            print("System: Please enter a message (or type 'quit').\n")
            continue

        risk = safety_risk(user_text, vec, clf)
        print(f"\nSafety risk (P(violation)): {risk:.2f}")

        rewrites = []
        if risk >= RISK_THRESHOLD:
            rewrites = propose_rewrites(user_text, use_case)
            print("Suggested rewrites:")
            for i, r in enumerate(rewrites, 1):
                print(f"  {i}) {r}")

        reply = partner_reply(user_text, persona, use_case)
        print(f"\n{persona.get('name','Partner')}: {reply}\n")

        episode["turns"].append({
            "turn": t,
            "user_text": user_text,
            "safe_risk": risk,
            "risk_threshold": RISK_THRESHOLD,
            "rewrites": rewrites,
            "partner_reply": reply
        })

    append_jsonl(OUT_LOG, episode)
    print(f"Saved episode to: {OUT_LOG}\n")


if __name__ == "__main__":
    main()
