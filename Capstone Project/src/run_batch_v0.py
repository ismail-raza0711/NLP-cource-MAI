#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS_DIR = DATA / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUBRIC_KEYS = ["ENG", "CTX", "TONE", "CLAR", "SAFE", "MOVE"]


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


# -------------------------
# v0 heuristic rubric scoring
# -------------------------
def score_message(user_text: str, persona: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, int]:
    t = user_text.strip()
    tl = t.lower()

    # CLAR: length heuristic
    if len(t) == 0 or len(t) > 250:
        clar = 0
    elif len(t) < 12:
        clar = 1
    else:
        clar = 2

    # ENG: question mark or interrogatives
    eng = 2 if ("?" in t or any(w in tl.split() for w in ["what", "how", "why", "where", "when"])) else (1 if len(t) >= 12 else 0)

    # CTX: references persona name or interests or profile cues
    ctx = 0
    if persona.get("name", "").lower() and persona["name"].lower() in tl:
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
    unsafe_markers = ["come over tonight", "send nudes", "don't be shy", "you owe", "why not", "stop being", "sensitive", "where do you live exactly"]
    if any(x in tl for x in unsafe_markers):
        safe = 0
    else:
        if context.get("use_case") == "UC4_BOUNDARY" and any(x in tl for x in ["fair", "no worries", "take it slow", "comfortable", "all good"]):
            safe = 2
        else:
            safe = 1

    # MOVE: depends on use case
    uc = context.get("use_case", "UNKNOWN")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate samples using v0 heuristic rubric scoring.")
    parser.add_argument("--personas", default=str(DATA / "personas.json"), help="Path to personas.json")
    parser.add_argument("--contexts", default=str(DATA / "contexts.jsonl"), help="Path to contexts.jsonl")
    parser.add_argument("--samples", default=str(DATA / "samples_unlabeled.jsonl"), help="Path to samples_unlabeled.jsonl")
    parser.add_argument("--out", default=str(RESULTS_DIR / "v0_batch_results.jsonl"), help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = no limit)")
    parser.add_argument("--seed", type=int, default=7, help="Reserved for future deterministic sampling; v0 ignores this.")
    args = parser.parse_args()

    personas = read_json(Path(args.personas))
    contexts = read_jsonl(Path(args.contexts))
    samples = read_jsonl(Path(args.samples))

    persona_by_id = {p["persona_id"]: p for p in personas}
    context_by_id = {c["context_id"]: c for c in contexts}

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    run_id = f"batch_{int(time.time())}"
    rows_out = []

    for s in samples:
        sample_id = s["sample_id"]
        context_id = s["context_id"]
        user_text = s["user_text"]
        use_case = s.get("use_case", "UNKNOWN")

        ctx = context_by_id.get(context_id)
        if ctx is None:
            # If contexts are missing, still record a row with errors
            rows_out.append({
                "run_id": run_id,
                "sample_id": sample_id,
                "context_id": context_id,
                "use_case": use_case,
                "error": f"Missing context_id={context_id}"
            })
            continue

        persona_id = ctx["persona_id"]
        persona = persona_by_id.get(persona_id)
        if persona is None:
            rows_out.append({
                "run_id": run_id,
                "sample_id": sample_id,
                "context_id": context_id,
                "use_case": use_case,
                "persona_id": persona_id,
                "error": f"Missing persona_id={persona_id}"
            })
            continue

        scores = score_message(user_text, persona, ctx)
        ocq_val = ocq(scores)
        safe_violation = 1 if scores["SAFE"] == 0 else 0

        rows_out.append({
            "run_id": run_id,
            "sample_id": sample_id,
            "context_id": context_id,
            "use_case": ctx.get("use_case", use_case),
            "persona_id": persona_id,
            "user_text": user_text,
            "scores": scores,
            "ocq": ocq_val,
            "safe_violation": safe_violation,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    out_path = Path(args.out)
    write_jsonl(out_path, rows_out)
    print(f"Wrote {len(rows_out)} rows to: {out_path}")

    # Quick summary of obvious data issues
    errors = [r for r in rows_out if "error" in r]
    if errors:
        print(f"WARNING: {len(errors)} rows contain errors (missing context/persona). First error:")
        print(json.dumps(errors[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
