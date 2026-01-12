#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(DATA/"samples_unlabeled.jsonl"))
    ap.add_argument("--out", default=str(DATA/"labels_safe_move_gold_template.jsonl"))
    ap.add_argument("--use_cases", nargs="+", default=["UC3_SUGGEST_DATE","UC4_BOUNDARY"])
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    samples = read_jsonl(Path(args.samples))
    filtered = [s for s in samples if s.get("use_case") in args.use_cases][:args.limit]

    rows = []
    for s in filtered:
        rows.append({
            "sample_id": s["sample_id"],
            "use_case": s.get("use_case",""),
            "category": s.get("category",""),
            "user_text": s.get("user_text",""),
            "SAFE": None,
            "MOVE": None,
            "notes": ""
        })

    write_jsonl(Path(args.out), rows)
    print(f"Wrote {len(rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
