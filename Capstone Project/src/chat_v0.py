#!/usr/bin/env python3
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = DATA / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

RUBRIC_KEYS = ["ENG", "CTX", "TONE", "CLAR", "SAFE", "MOVE"]

# --- IO helpers ---
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl_append(path: Path, row: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# --- Heuristic rubric scoring (v0 baseline) ---
def score_message(user_text: str, persona: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, int]:
    t = user_text.strip()
    tl = t.lower()

    # CLAR: simple length + punctuation heuristic
    if len(t) == 0 or len(t) > 250:
        clar = 0
    elif len(t) < 12:
        clar = 1
    else:
        clar = 2

    # ENG: question mark or interrogatives
    eng = 2 if ("?" in t or any(w in tl.split() for w in ["what", "how", "why", "where", "when"])) else (1 if len(t) >= 12 else 0)

    # CTX: references persona name or interests or mentions Berlin/profile cues
    ctx = 0
    if persona["name"].lower() in tl:
        ctx = 2
    else:
        hits = 0
        for it in persona.get("interests", []):
            if it.lower() in tl:
                hits += 1
        if hits >= 1:
            ctx = 2
        elif any(k in tl for k in ["berlin", "your bio", "profile"]):
            ctx = 1

    # TONE: crude detection
    rude_markers = ["whatever", "shut up", "idiot", "stupid", "bitch"]
    objectifying = ["hot", "sexy", "tits"]
    if any(x in tl for x in rude_markers) or any(x in tl for x in objectifying):
        tone = 0
    elif any(x in tl for x in ["please", "thanks", "no worries", "fair", "totally"]):
        tone = 2
    else:
        tone = 1

    # SAFE: boundary pressure / coercion heuristics
    unsafe_markers = ["come over tonight", "send nudes", "don't be shy", "you owe", "why not", "stop being", "sensitive"]
    if any(x in tl for x in unsafe_markers):
        safe = 0
    else:
        # if in boundary use case, reward explicit respect
        if context["use_case"] == "UC4_BOUNDARY" and any(x in tl for x in ["fair", "no worries", "take it slow", "comfortable", "all good"]):
            safe = 2
        else:
            safe = 1

    # MOVE: depends on use case
    uc = context["use_case"]
    if uc == "UC1_COLD_OPEN":
        move = 2 if eng >= 1 else 1
    elif uc == "UC2_KEEP_GOING":
        move = 2 if eng >= 1 and len(t) >= 12 else 1 if len(t) >= 8 else 0
    elif uc == "UC3_SUGGEST_DATE":
        move = 2 if any(x in tl for x in ["coffee", "walk", "meet", "grab a"]) and safe != 0 else 1
    elif uc == "UC4_BOUNDARY":
        move = 2 if safe == 2 else 0 if safe == 0 else 1
    else:
        move = 1

    return {"ENG": eng, "CTX": ctx, "TONE": tone, "CLAR": clar, "SAFE": safe, "MOVE": move}

def ocq(scores: Dict[str, int]) -> float:
    total = sum(scores[k] for k in RUBRIC_KEYS)  # 0..12
    return total / 12.0

def feedback_from_scores(scores: Dict[str, int], persona: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    fb = []
    if scores["ENG"] <= 1:
        fb.append("Add an engaging hook and a concrete question to invite a reply.")
    if scores["CTX"] <= 1:
        fb.append(f"Reference something specific from the profile (e.g., {persona['interests'][0]}).")
    if scores["TONE"] == 0:
        fb.append("Adjust tone to be respectful and non-objectifying.")
    if scores["CLAR"] <= 1:
        fb.append("Make the message clearer and slightly more specific; aim for 1–2 short sentences.")
    if context["use_case"] == "UC4_BOUNDARY" and scores["SAFE"] <= 1:
        fb.append("Acknowledge the boundary and explicitly agree to slow down.")
    if not fb:
        fb.append("Solid message. Consider a slightly more personal detail to deepen the conversation.")
    return fb[:3]

# --- Partner reply generator (rule-based) ---
def partner_reply(user_text: str, persona: Dict[str, Any], context: Dict[str, Any]) -> str:
    tl = user_text.lower()
    style = persona.get("style", "neutral")
    uc = context["use_case"]
    interests = persona.get("interests", [])

    # Minimal safety reaction
    if any(x in tl for x in ["send nudes", "come over tonight", "you owe", "where do you live exactly"]):
        return "I’m not comfortable with that. Let’s keep it respectful."

    if uc == "UC1_COLD_OPEN":
        if "?" in user_text:
            base = f"{'Haha' if style=='warm' else ''} I’m into {random.choice(interests)} lately. What about you?"
        else:
            base = "Hey! What’s up?"
        return base.strip()

    if uc == "UC2_KEEP_GOING":
        if "week" in tl or "busy" in tl:
            return "Sounds like a full week. Anything fun planned besides work?"
        if "?" in user_text:
            return "Good question. I’m mostly keeping it simple—maybe a ride and some music."
        return "Got it. Tell me a bit more."

    if uc == "UC3_SUGGEST_DATE":
        if any(x in tl for x in ["coffee", "walk", "meet", "grab a"]):
            return "That sounds nice. I’d be up for a coffee—what day works for you?"
        return "Maybe—what kind of plan did you have in mind?"

    if uc == "UC4_BOUNDARY":
        if any(x in tl for x in ["no worries", "fair", "take it slow", "comfortable"]):
            return "Thanks for understanding. Let’s just chat and see how it goes."
        return "I appreciate it if we slow down a bit."

    return "Okay."

def main():
    personas = read_json(DATA / "personas.json")
    contexts = read_jsonl(DATA / "contexts.jsonl")
    persona_by_id = {p["persona_id"]: p for p in personas}

    # Choose context
    context = random.choice(contexts)
    persona = persona_by_id[context["persona_id"]]

    print("\n=== Tinder Practice Bot v0 (baseline, rule-based) ===\n")
    print(f"Use case: {context['use_case']}")
    print(f"Partner persona: {persona['name']}")
    print(f"Bio: {persona['bio']}\n")

    # Print initial partner message if present
    history = list(context.get("chat_history", []))
    if context.get("partner_last_message"):
        print(f"{persona['name']}: {context['partner_last_message']}\n")

    max_user_turns = 4
    episode = {
        "episode_id": f"ep_{int(time.time())}_{random.randint(1000,9999)}",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "context_id": context["context_id"],
        "use_case": context["use_case"],
        "persona_id": persona["persona_id"],
        "turns": []
    }

    for turn_idx in range(max_user_turns):
        user_text = input("You: ").strip()
        if user_text.lower() in ["q", "quit", "exit"]:
            break

        scores = score_message(user_text, persona, context)
        ocq_val = ocq(scores)
        fb = feedback_from_scores(scores, persona, context)

        reply = partner_reply(user_text, persona, context)

        print(f"\nCoach scores: {scores}  OCQ={ocq_val:.2f}")
        for i, tip in enumerate(fb, 1):
            print(f"Coach tip {i}: {tip}")
        print(f"\n{persona['name']}: {reply}\n")

        episode["turns"].append({
            "turn": turn_idx + 1,
            "user_text": user_text,
            "scores": scores,
            "ocq": ocq_val,
            "coach_tips": fb,
            "partner_reply": reply
        })

    write_jsonl_append(LOGS / "episodes.jsonl", episode)
    print(f"\nSaved episode to {LOGS / 'episodes.jsonl'}")

if __name__ == "__main__":
    main()
