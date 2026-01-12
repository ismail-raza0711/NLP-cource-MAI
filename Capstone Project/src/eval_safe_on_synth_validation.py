#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = DATA / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DEFAULT_VALIDATION = DATA / "labels_safe_move_synth_validation.jsonl"
DEFAULT_MODEL = MODELS / "safe_violation_clf.joblib"
DEFAULT_OUT = RESULTS / "v0_3_synth_validation_report.json"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def confusion_counts(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    # positive class = violation = 1
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def prf(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    ap = argparse.ArgumentParser(description="Evaluate v0.2 safety classifier on synthetic validation set.")
    ap.add_argument("--validation", default=str(DEFAULT_VALIDATION), help="JSONL with user_text + SAFE labels")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="joblib bundle: {vectorizer, model}")
    ap.add_argument("--threshold", type=float, default=0.35, help="risk threshold for violation prediction")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="write JSON report here")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    vec = bundle["vectorizer"]
    clf = bundle["model"]

    rows = read_jsonl(Path(args.validation))

    # ground truth: violation if SAFE == 0
    y_true = []
    y_pred = []
    by_uc = defaultdict(lambda: {"y_true": [], "y_pred": []})
    examples = []

    for r in rows:
        text = r.get("user_text", "")
        use_case = r.get("use_case", "UNKNOWN")
        safe = r.get("SAFE", None)

        if safe is None:
            continue

        true_violation = 1 if int(safe) == 0 else 0
        risk = float(clf.predict_proba(vec.transform([text]))[0][1])
        pred_violation = 1 if risk >= args.threshold else 0

        y_true.append(true_violation)
        y_pred.append(pred_violation)
        by_uc[use_case]["y_true"].append(true_violation)
        by_uc[use_case]["y_pred"].append(pred_violation)

        # store a few error examples for debugging/reporting
        if len(examples) < 20 and true_violation != pred_violation:
            examples.append({
                "sample_id": r.get("sample_id"),
                "use_case": use_case,
                "category": r.get("category", ""),
                "user_text": text,
                "SAFE": safe,
                "risk": risk,
                "pred_violation": pred_violation
            })

    overall_counts = confusion_counts(y_true, y_pred)
    overall_prf = prf(overall_counts)

    per_uc = {}
    for uc, d in by_uc.items():
        c = confusion_counts(d["y_true"], d["y_pred"])
        per_uc[uc] = {
            "n": len(d["y_true"]),
            "confusion": c,
            "metrics": prf(c),
        }

    report = {
        "validation_file": str(Path(args.validation)),
        "model_file": str(Path(args.model)),
        "threshold": args.threshold,
        "n_scored": len(y_true),
        "overall": {
            "confusion": overall_counts,
            "metrics": overall_prf,
            "accuracy": sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
        },
        "by_use_case": per_uc,
        "error_examples": examples,
        "notes": "Ground truth defined as violation iff SAFE==0 (synthetic validation)."
    }

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== v0.3 synth validation report ===")
    print(f"Scored n={report['n_scored']} rows at threshold={args.threshold}")
    print("Overall confusion:", report["overall"]["confusion"])
    m = report["overall"]["metrics"]
    print(f"Precision={m['precision']:.3f} Recall={m['recall']:.3f} F1={m['f1']:.3f} Acc={report['overall']['accuracy']:.3f}")
    print(f"Wrote report to: {args.out}")

    # short per-UC summary
    for uc, d in report["by_use_case"].items():
        mm = d["metrics"]
        print(f"- {uc}: n={d['n']}, P={mm['precision']:.3f}, R={mm['recall']:.3f}, F1={mm['f1']:.3f}")


if __name__ == "__main__":
    main()
