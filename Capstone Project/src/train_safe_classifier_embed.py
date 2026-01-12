# src/train_safe_classifier_embed.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def derive_y_from_scores_threshold(
    r: Dict[str, Any],
    safe_key: str,
    move_key: str,
    move_threshold: int,
) -> Optional[int]:
    """
    y=1 (MOVE) if MOVE score >= move_threshold, else SAFE (0).
    """
    if safe_key not in r or move_key not in r:
        return None

    # validate both fields exist and are ints
    _safe_v = parse_int(r.get(safe_key), safe_key)
    move_v = parse_int(r.get(move_key), move_key)

    return 1 if move_v >= move_threshold else 0


def extract_xy(
    rows: List[Dict[str, Any]],
    text_key: str,
    safe_key: str,
    move_key: str,
    move_threshold: int,
) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    y: List[int] = []
    skipped_other = 0

    for r in rows:
        t = r.get(text_key, "")
        if not isinstance(t, str):
            skipped_other += 1
            continue
        t = t.strip()
        if not t:
            skipped_other += 1
            continue

        yi = derive_y_from_scores_threshold(
            r,
            safe_key=safe_key,
            move_key=move_key,
            move_threshold=move_threshold,
        )
        if yi is None:
            skipped_other += 1
            continue

        texts.append(t)
        y.append(int(yi))

    if len(texts) < 20:
        raise ValueError(f"Too few usable rows after filtering: {len(texts)} (skipped other={skipped_other}).")

    y_arr = np.array(y, dtype=np.int32)
    n_move = int(y_arr.sum())
    n_safe = int(len(y_arr) - n_move)
    print(f"[INFO] After mapping: n={len(y_arr)} | SAFE={n_safe} | MOVE={n_move} | move_threshold={move_threshold}")

    uniq = set(y_arr.tolist())
    if uniq == {0} or uniq == {1}:
        raise ValueError(
            f"Training data is single-class after mapping (classes present: {sorted(uniq)}). "
            f"Adjust --move_threshold or generate more diverse synthetic data."
        )

    return texts, y_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="data/labels_safe_move_synth_validation.jsonl")
    ap.add_argument("--text_key", default="user_text")
    ap.add_argument("--safe_key", default="SAFE")
    ap.add_argument("--move_key", default="MOVE")
    ap.add_argument("--move_threshold", type=int, default=2, help="Label as MOVE if MOVE score >= this value")
    ap.add_argument("--out_model", default="models/safe_violation_clf_embed.joblib")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_iter", type=int, default=2000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path = Path(args.train_jsonl)
    rows = read_jsonl(train_path)
    if not rows:
        raise ValueError(f"Training file appears empty: {train_path}")

    texts, y = extract_xy(
        rows,
        text_key=args.text_key,
        safe_key=args.safe_key,
        move_key=args.move_key,
        move_threshold=args.move_threshold,
    )

    embedder = SentenceTransformer(args.embed_model)
    X = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    classes = np.array([0, 1], dtype=np.int32)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}

    clf = LogisticRegression(
        random_state=args.seed,
        max_iter=args.max_iter,
        C=args.C,
        class_weight=class_weight,
        solver="lbfgs",
    )
    clf.fit(X, y)

    artifact = {
        "type": "embed_lr",
        "sentence_transformer": args.embed_model,
        "normalize_embeddings": True,
        "text_key": args.text_key,
        "safe_key": args.safe_key,
        "move_key": args.move_key,
        "move_threshold": args.move_threshold,
        "label_mapping": {"SAFE": 0, "MOVE": 1},
        "class_weight": class_weight,
        "train_source": str(train_path),
        "n_train": int(len(texts)),
        "n_move": int(y.sum()),
        "seed": args.seed,
        "C": args.C,
        "max_iter": args.max_iter,
        "logreg": clf,
    }

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)

    print(f"[OK] Saved embedding model to: {out_path}")


if __name__ == "__main__":
    main()
