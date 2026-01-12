#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "data" / "results" / "v0_batch_results.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Report aggregate metrics from batch results JSONL.")
    parser.add_argument("--in", dest="in_path", default=str(DEFAULT_RESULTS), help="Path to batch results JSONL")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    rows = read_jsonl(in_path)

    # filter out error rows or incomplete rows
    scored = [r for r in rows if "error" not in r and isinstance(r.get("ocq"), (int, float))]

    if not scored:
        print(f"No scored rows found in {in_path}.")
        return

    ocqs = [r["ocq"] for r in scored]
    viols = [r.get("safe_violation", 0) for r in scored]

    print(f"Input file: {in_path}")
    print(f"Scored rows: {len(scored)} / {len(rows)}")
    print(f"Mean OCQ:   {mean(ocqs):.3f}")
    print(f"Median OCQ: {median(ocqs):.3f}")
    print(f"Safety violation rate: {100.0 * mean(viols):.1f}%")

    # By use case
    by_uc: Dict[str, Dict[str, List[float]]] = {}
    for r in scored:
        uc = r.get("use_case", "UNKNOWN")
        by_uc.setdefault(uc, {"ocq": [], "viol": []})
        by_uc[uc]["ocq"].append(r["ocq"])
        by_uc[uc]["viol"].append(r.get("safe_violation", 0))

    print("\nBy use case:")
    for uc in sorted(by_uc.keys()):
        ocq_list = by_uc[uc]["ocq"]
        v_list = by_uc[uc]["viol"]
        print(f"- {uc}: n={len(ocq_list)}, mean_OCQ={mean(ocq_list):.3f}, viol%={100.0*mean(v_list):.1f}")


if __name__ == "__main__":
    main()
