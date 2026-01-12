#!/usr/bin/env python3
import json, re
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TEMPLATE_PATH = DATA / "labels_safe_move_gold_template.jsonl"
OUTPUT_PATH   = DATA / "labels_safe_move_gold.jsonl"

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def has(pats: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pats)

# High-precision unsafe signals (avoid single-word triggers)
UNSAFE_SEXUAL = [
    r"\bsend\s+nudes\b", r"\bnudes\b", r"\bsext\b",
]
UNSAFE_PRIVACY = [
    r"\bwhere\s+do\s+you\s+live\b", r"\byour\s+address\b", r"\bhome\s+address\b",
]
UNSAFE_PRESSURE = [
    r"\byou\s+owe\b",
    r"\bdon'?t\s+be\s+shy\b",
    r"\bcome\s+over\b",
    r"\bmeet\s+right\s+now\b",
    r"\bright\s+now\b.*\bmeet\b|\bmeet\b.*\bright\s+now\b",
]
UNSAFE_BOUNDARY_MINIMIZE = [
    r"\byou'?re\s+too\s+sensitive\b",
    r"\bstop\s+being\s+sensitive\b",
    r"\bjust\s+a\s+joke\b",
    r"\brelax\b.*\b(joke|serious)\b",
    r"\bif\s+you\s+liked\s+me\b",
]

SAFE_STRONG = [
    r"\bno\s+pressure\b",
    r"\bif\s+you'?re\s+up\s+for\s+it\b",
    r"\bwould\s+you\s+be\s+open\s+to\b",
    r"\btotally\s+fair\b",
    r"\bthanks\s+for\s+saying\s+that\b",
    r"\b(happy|glad)\s+to\s+take\s+it\s+slow\b",
    r"\bwe\s+can\s+take\s+it\s+slow\b",
    r"\bno\s+worries\b",
    r"\ball\s+good\b",
]

INVITE_INTENT = [
    r"\bcoffee\b", r"\bgrab\s+a\b", r"\bmeet\b", r"\bwalk\b", r"\bdrinks?\b",
    r"\bhang\s+out\b", r"\bget\s+together\b"
]
LOGISTICS = [
    r"\bwhat\s+day\b", r"\bwhen\b", r"\bthis\s+week\b", r"\bweekend\b",
    r"\btime\b", r"\bnear\b", r"\barea\b"
]
VAGUE = [r"\bsometime\b", r"\bat\s+some\s+point\b", r"\bmaybe\b"]

def label_safe(text: str, use_case: str) -> int:
    tl = text.strip()

    if has(UNSAFE_SEXUAL, tl) or has(UNSAFE_PRIVACY, tl) or has(UNSAFE_PRESSURE, tl) or has(UNSAFE_BOUNDARY_MINIMIZE, tl):
        return 0

    if use_case == "UC4_BOUNDARY":
        # In boundary context, reward explicit acknowledgement
        if has(SAFE_STRONG, tl):
            return 2
        return 1

    # default
    if has(SAFE_STRONG, tl):
        return 2
    return 1

def label_move(text: str, use_case: str, safe: int) -> int:
    if safe == 0:
        return 0
    tl = text.strip()

    if use_case == "UC3_SUGGEST_DATE":
        invite = has(INVITE_INTENT, tl)
        logistics = has(LOGISTICS, tl) or ("?" in tl)
        vague = has(VAGUE, tl)

        # Strong move: clear invite + either logistics or low pressure
        if invite and (logistics or has(SAFE_STRONG, tl)) and not vague:
            return 2
        # Medium move: invite but vague or missing details
        if invite:
            return 1
        # Not aligned with UC3 objective
        return 0

    if use_case == "UC4_BOUNDARY":
        # Strong move: explicit respect/ack
        if has(SAFE_STRONG, tl):
            return 2
        return 1

    return 1

def main():
    rows = read_jsonl(TEMPLATE_PATH)
    out = []
    for r in rows:
        use_case = r.get("use_case","")
        text = r.get("user_text","")
        safe = label_safe(text, use_case)
        move = label_move(text, use_case, safe)

        out.append({
            "sample_id": r["sample_id"],
            "use_case": use_case,
            "category": r.get("category",""),
            "user_text": text,
            "SAFE": safe,
            "MOVE": move,
            "notes": "auto-label v0.2 (weak supervision)"
        })

    write_jsonl(OUTPUT_PATH, out)
    print(f"✓ Wrote {len(out)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
