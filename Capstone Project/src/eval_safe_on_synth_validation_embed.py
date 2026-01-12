# src/eval_safe_on_synth_validation_embed.py
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_int(v: Any, field: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"Could not parse '{field}' as int: {v}") from e


def y_true_from_scores(rows: List[Dict[str, Any]], safe_key: str, move_key: str, move_threshold: int) -> np.ndarray:
    y = []
    for r in rows:
        _ = parse_int(r.get(safe_key), safe_key)
        mv = parse_int(r.get(move_key), move_key)
        y.append(1 if mv >= move_threshold else 0)
    return np.array(y, dtype=np.int32)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def prf(conf: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = conf["tp"], conf["tn"], conf["fp"], conf["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def predict_p_move(model_artifact: Any, texts: List[str]) -> np.ndarray:
    """
    Supports your embedding artifact dict:
      {"type":"embed_lr", "sentence_transformer":..., "logreg":...}
    """
    if isinstance(model_artifact, dict) and model_artifact.get("type") == "embed_lr":
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(model_artifact["sentence_transformer"])
        X = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=bool(model_artifact.get("normalize_embeddings", True)),
        )
        clf = model_artifact["logreg"]
        return clf.predict_proba(X)[:, 1]

    # fallback if you ever pass a sklearn pipeline
    return model_artifact.predict_proba(texts)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/safe_violation_clf_embed.joblib")
    ap.add_argument("--in_path", default="data/labels_safe_move_synth_validation.jsonl")
    ap.add_argument("--out_path", default="data/results/v0_3_synth_validation_report_embed.json")
    ap.add_argument("--threshold", type=float, default=0.35, help="Decision threshold on predicted p(MOVE)")
    ap.add_argument("--text_key", default="user_text")
    ap.add_argument("--safe_key", default="SAFE")
    ap.add_argument("--move_key", default="MOVE")
    ap.add_argument("--move_threshold", type=int, default=2, help="Ground-truth mapping: MOVE if MOVE>=move_threshold")
    ap.add_argument("--uc_key", default="use_case")
    args = ap.parse_args()

    model = joblib.load(args.model_path)
    rows = read_jsonl(Path(args.in_path))
    if not rows:
        raise ValueError(f"No rows found in {args.in_path}")

    texts = [str(r.get(args.text_key, "")).strip() for r in rows]
    y_true = y_true_from_scores(rows, args.safe_key, args.move_key, args.move_threshold)

    p_move = predict_p_move(model, texts)
    y_pred = (p_move >= args.threshold).astype(np.int32)

    overall_conf = confusion_counts(y_true, y_pred)
    overall_metrics = prf(overall_conf)

    print("=== embedding synth validation report ===")
    print(f"Scored n={len(rows)} rows at threshold={args.threshold} | gt_move_threshold={args.move_threshold}")
    print(f"Overall confusion: {overall_conf}")
    print(
        f"Precision={overall_metrics['precision']:.3f} "
        f"Recall={overall_metrics['recall']:.3f} "
        f"F1={overall_metrics['f1']:.3f} "
        f"Acc={overall_metrics['accuracy']:.3f}"
    )

    per_uc = {}
    if args.uc_key in rows[0]:
        buckets = defaultdict(list)
        for r, yt, yp in zip(rows, y_true.tolist(), y_pred.tolist()):
            buckets[str(r.get(args.uc_key, "UNKNOWN"))].append((yt, yp))

        for uc, pairs in buckets.items():
            yt = np.array([p[0] for p in pairs], dtype=np.int32)
            yp = np.array([p[1] for p in pairs], dtype=np.int32)
            conf = confusion_counts(yt, yp)
            m = prf(conf)
            per_uc[uc] = {"n": int(len(pairs)), **m, "confusion": conf}
            print(f"- {uc}: n={len(pairs)}, P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}")

    report = {
        "version": "v0.3",
        "model_path": args.model_path,
        "in_path": args.in_path,
        "threshold": args.threshold,
        "gt_move_threshold": args.move_threshold,
        "overall": {"n": int(len(rows)), **overall_metrics, "confusion": overall_conf},
        "per_use_case": per_uc,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report to: {out_path}")


if __name__ == "__main__":
    main()
