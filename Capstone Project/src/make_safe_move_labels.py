#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "labels_safe_move_gold_template.jsonl"))
    ap.add_argument("--use_cases", nargs="+", default=["UC3_SUGGEST_DATE", "UC4_BOUNDARY"])
    ap.add_argument("--limit", type=int, default=300, help="How many samples to include for labeling.")
    args = ap.parse_args()

    samples = read_jsonl(DATA / "samples_unlabeled.jsonl")
    filtered = [s for s in samples if s.get("use_case") in args.use_cases]
    filtered = filtered[: args.limit]

    labels = []
    for s in filtered:
        labels.append({
            "sample_id": s["sample_id"],
            "use_case": s.get("use_case"),
            "category": s.get("category", ""),
            "SAFE": None,   # 0/1/2
            "MOVE": None,   # 0/1/2
            "notes": ""
        })

    write_jsonl(Path(args.out), labels)
    print(f"Wrote {len(labels)} label rows to {args.out}")

if __name__ == "__main__":
    main()
