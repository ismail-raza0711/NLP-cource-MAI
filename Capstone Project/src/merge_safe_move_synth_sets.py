# src/merge_safe_move_synth_sets.py
import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedupe_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    out: List[Dict[str, object]] = []
    for r in rows:
        sid = r.get("sample_id")
        key = sid if sid else json.dumps(r, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/labels_safe_move_synth_validation.jsonl")
    ap.add_argument("--safe_expansion", default="data/labels_safe_move_synth_safe_expansion.jsonl")
    ap.add_argument("--out", default="data/labels_safe_move_synth_merged.jsonl")
    args = ap.parse_args()

    base_rows = read_jsonl(Path(args.base))
    exp_rows = read_jsonl(Path(args.safe_expansion))
    merged = dedupe_rows(base_rows + exp_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[OK] Wrote {len(merged)} rows to {out_path}")


if __name__ == "__main__":
    main()
