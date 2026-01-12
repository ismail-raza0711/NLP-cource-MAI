#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

UC3_BANK = [
    # good / low pressure
    ("low_pressure_invite", "Low-key sounds nice. Want to grab a coffee sometime this week? No pressure."),
    ("low_pressure_invite", "If you’re up for it, we could do a short walk and a coffee—totally chill."),
    ("logistics_friendly", "Would you be open to meeting for coffee? I’m flexible on day/time."),
    ("logistics_friendly", "We could meet near a central spot—any area you prefer?"),

    # meh
    ("too_vague", "We should hang out sometime."),
    ("too_vague", "Maybe we can meet at some point."),

    # unsafe/pushy
    ("pushy_escalation", "Come over tonight."),
    ("pushy_escalation", "Let’s meet right now. Don’t be shy."),
    ("privacy_invasive", "Let’s meet. Where do you live exactly?"),
    ("privacy_invasive", "Send me your address and I’ll come by."),
]

UC4_BANK = [
    # good boundary handling
    ("ack_backoff", "Totally fair—thanks for saying that. I’m happy to take it slow."),
    ("ack_backoff", "No worries at all. We can just keep chatting and see how it goes."),
    ("repair_shift", "Got it. Let’s slow down—what kind of music are you into?"),

    # meh
    ("awk_ack", "Okay, sorry. Anyway…"),
    ("awk_ack", "Alright, we can slow down."),

    # unsafe/minimizing
    ("minimizing", "You’re being too sensitive."),
    ("minimizing", "Relax, it was just a joke."),
    ("guilt", "Wow, I thought you liked me…"),
    ("guilt", "If you were into me, you wouldn’t say that."),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add_per_uc", type=int, default=125, help="How many additional samples to add per UC (UC3 and UC4).")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)

    personas = read_json(DATA / "personas.json")
    contexts = read_jsonl(DATA / "contexts.jsonl")
    samples = read_jsonl(DATA / "samples_unlabeled.jsonl")

    context_by_uc = {}
    for c in contexts:
        context_by_uc.setdefault(c["use_case"], []).append(c)

    # Determine next sample_id
    def parse_id(sid: str) -> int:
        # expects s000001
        try:
            return int(sid[1:])
        except Exception:
            return 0

    max_id = max((parse_id(s["sample_id"]) for s in samples), default=0)
    next_id = max_id + 1

    new_samples = []

    def add_samples_for_uc(use_case: str, bank: List, n: int):
        nonlocal next_id
        ctx_list = context_by_uc.get(use_case, [])
        if not ctx_list:
            raise RuntimeError(f"No contexts found for {use_case}")

        for _ in range(n):
            ctx = random.choice(ctx_list)
            cat, text = random.choice(bank)
            sid = f"s{next_id:06d}"
            next_id += 1
            new_samples.append({
                "sample_id": sid,
                "context_id": ctx["context_id"],
                "use_case": use_case,
                "category": cat,
                "user_text": text
            })

    add_samples_for_uc("UC3_SUGGEST_DATE", UC3_BANK, args.add_per_uc)
    add_samples_for_uc("UC4_BOUNDARY", UC4_BANK, args.add_per_uc)

    merged = samples + new_samples
    write_jsonl(DATA / "samples_unlabeled.jsonl", merged)

    print(f"Added {len(new_samples)} samples. Total samples now: {len(merged)}")
    print("Note: new samples include a 'category' field for analysis.")

if __name__ == "__main__":
    main()
