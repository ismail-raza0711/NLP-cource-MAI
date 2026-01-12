#!/usr/bin/env python3
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

USE_CASES = ["UC1_COLD_OPEN", "UC2_KEEP_GOING", "UC3_SUGGEST_DATE", "UC4_BOUNDARY"]

PERSONAS = [
    {
        "persona_id": "p001",
        "name": "Mira",
        "bio": "Berlin-based product designer. Loves bouldering and museums. Coffee snob.",
        "interests": ["bouldering", "museums", "coffee", "design"],
        "style": "warm",
    },
    {
        "persona_id": "p002",
        "name": "Jonas",
        "bio": "PhD student into techno, cycling, and street photography. Introvert until espresso.",
        "interests": ["techno", "cycling", "photography", "espresso"],
        "style": "dry",
    },
]

CONTEXTS = [
    {"context_id": "c001", "use_case": "UC1_COLD_OPEN", "persona_id": "p001",
     "chat_history": [], "partner_last_message": None},
    {"context_id": "c002", "use_case": "UC2_KEEP_GOING", "persona_id": "p002",
     "chat_history": [{"speaker": "partner", "text": "Nice to meet you! What are you up to this week?"}],
     "partner_last_message": "Nice to meet you! What are you up to this week?"},
    {"context_id": "c003", "use_case": "UC3_SUGGEST_DATE", "persona_id": "p001",
     "chat_history": [{"speaker": "partner", "text": "I’m usually free after work, but I like keeping it low-key."}],
     "partner_last_message": "I’m usually free after work, but I like keeping it low-key."},
    {"context_id": "c004", "use_case": "UC4_BOUNDARY", "persona_id": "p002",
     "chat_history": [{"speaker": "partner", "text": "Hey, that’s a bit fast for me. I prefer to take it slow."}],
     "partner_last_message": "Hey, that’s a bit fast for me. I prefer to take it slow."},
]

# Simple message banks to create a starter unlabeled dataset.
# These are intentionally mixed-quality to support rubric scoring later.
MESSAGE_BANK = {
    "UC1_COLD_OPEN": [
        "Hey {name} — {interest1} + {interest2} is a strong combo. What’s your go-to spot in Berlin?",
        "Hi {name}.",
        "Your bio is cool. What do you do?",
        "If you had to pick one: {interest1} or {interest2}?",
        "Wow you’re hot.",
    ],
    "UC2_KEEP_GOING": [
        "This week is a bit busy, but I’m trying to squeeze in {interest1}. How about you?",
        "Not much lol",
        "Probably work and sleep. You?",
        "I’m free on Thursday evening—what do you like doing after work?",
        "Why do you even ask that?",
    ],
    "UC3_SUGGEST_DATE": [
        "Low-key sounds great. Want to grab a coffee near {interest1}-friendly area sometime this week?",
        "Come over tonight.",
        "If you’re up for it, we could do a short walk and coffee—no pressure.",
        "Let’s meet. Where do you live exactly?",
        "We should totally hang out, maybe.",
    ],
    "UC4_BOUNDARY": [
        "Totally fair—thanks for saying that. I’m happy to take it slow. What pace feels comfortable for you?",
        "Relax, it was just a joke.",
        "Okay whatever.",
        "No worries at all. We can just chat and see if we vibe.",
        "You’re being too sensitive.",
    ],
}

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main(n_per_use_case: int = 25, seed: int = 7):
    random.seed(seed)
    DATA.mkdir(parents=True, exist_ok=True)

    # Write personas
    write_json(DATA / "personas.json", PERSONAS)

    # Write contexts
    write_jsonl(DATA / "contexts.jsonl", CONTEXTS)

    persona_by_id = {p["persona_id"]: p for p in PERSONAS}
    contexts_by_use = {}
    for c in CONTEXTS:
        contexts_by_use.setdefault(c["use_case"], []).append(c)

    samples = []
    labels = []
    sample_counter = 1

    for uc in USE_CASES:
        for _ in range(n_per_use_case):
            ctx = random.choice(contexts_by_use[uc])
            persona = persona_by_id[ctx["persona_id"]]

            template = random.choice(MESSAGE_BANK[uc])
            interest1, interest2 = random.sample(persona["interests"], k=2)
            user_text = template.format(
                name=persona["name"],
                interest1=interest1,
                interest2=interest2,
            )

            sample_id = f"s{sample_counter:06d}"
            sample_counter += 1

            samples.append({
                "sample_id": sample_id,
                "context_id": ctx["context_id"],
                "use_case": uc,
                "user_text": user_text
            })

            labels.append({
                "sample_id": sample_id,
                "ENG": None, "CTX": None, "TONE": None,
                "CLAR": None, "SAFE": None, "MOVE": None,
                "notes": ""
            })

    write_jsonl(DATA / "samples_unlabeled.jsonl", samples)
    write_jsonl(DATA / "labels_template.jsonl", labels)

    # Create a tiny gold example for demonstration (label a few obvious ones)
    gold = []
    for s in samples[:5]:
        # Heuristic: label the first few as "good" only if they aren't obviously bad
        txt = s["user_text"].lower()
        if any(bad in txt for bad in ["hot", "come over", "sensitive", "whatever", "why do you even"]):
            continue
        gold.append({
            "sample_id": s["sample_id"],
            "ENG": 2, "CTX": 2, "TONE": 2, "CLAR": 2, "SAFE": 2, "MOVE": 2,
            "notes": "Autofilled example label for documentation/demo."
        })
        if len(gold) >= 3:
            break
    write_jsonl(DATA / "labels_gold_example.jsonl", gold)

    print("Wrote:")
    print(" - data/personas.json")
    print(" - data/contexts.jsonl")
    print(" - data/samples_unlabeled.jsonl")
    print(" - data/labels_template.jsonl")
    print(" - data/labels_gold_example.jsonl")

if __name__ == "__main__":
    main()
