#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RUBRIC_KEYS = ["ENG", "CTX", "TONE", "CLAR", "SAFE", "MOVE"]

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def ocq(label_row):
    vals = [label_row[k] for k in RUBRIC_KEYS]
    if any(v is None for v in vals):
        return None
    total = sum(vals)  # 0..12
    return total / 12.0

def main(labels_file="labels_gold_example.jsonl", samples_file="samples_unlabeled.jsonl"):
    labels = read_jsonl(DATA / labels_file)
    samples = read_jsonl(DATA / samples_file)
    sample_by_id = {s["sample_id"]: s for s in samples}

    scored = []
    for r in labels:
        score = ocq(r)
        if score is None:
            continue
        use_case = sample_by_id.get(r["sample_id"], {}).get("use_case", "UNKNOWN")
        safe_violation = 1 if r["SAFE"] == 0 else 0
        scored.append((use_case, score, safe_violation))

    if not scored:
        print("No fully-scored labels found. Use a labeled file where rubric values are not null.")
        return

    overall_scores = [s for _, s, _ in scored]
    overall_viol = [v for _, _, v in scored]

    print(f"Scored messages: {len(scored)}")
    print(f"Mean OCQ: {mean(overall_scores):.3f}")
    print(f"Median OCQ: {median(overall_scores):.3f}")
    print(f"Safety violation rate: {100.0 * mean(overall_viol):.1f}%")

    # By use-case
    by_uc = {}
    for uc, sc, v in scored:
        by_uc.setdefault(uc, {"scores": [], "viol": []})
        by_uc[uc]["scores"].append(sc)
        by_uc[uc]["viol"].append(v)

    print("\nBy use case:")
    for uc, d in sorted(by_uc.items()):
        print(f"- {uc}: n={len(d['scores'])}, mean_OCQ={mean(d['scores']):.3f}, viol%={100.0*mean(d['viol']):.1f}")

if __name__ == "__main__":
    main()
