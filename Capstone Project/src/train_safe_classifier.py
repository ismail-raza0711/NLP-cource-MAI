#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(DATA / "labels_safe_move_gold.jsonl"))
    ap.add_argument("--samples", default=str(DATA / "samples_unlabeled.jsonl"))
    ap.add_argument("--out", default=str(MODELS / "safe_violation_clf.joblib"))
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    labels = read_jsonl(Path(args.labels))
    samples = read_jsonl(Path(args.samples))
    sample_by_id = {s["sample_id"]: s for s in samples}

    X_text = []
    y = []

    skipped = 0
    for r in labels:
        sid = r["sample_id"]
        s = sample_by_id.get(sid)
        if s is None:
            skipped += 1
            continue
        SAFE = r.get("SAFE")
        if SAFE is None:
            skipped += 1
            continue
        # binary target
        safe_violation = 1 if SAFE == 0 else 0
        X_text.append(s["user_text"])
        y.append(safe_violation)

    if len(X_text) < 50:
        raise RuntimeError(f"Not enough labeled samples to train (got {len(X_text)}). Fill labels_safe_move_gold.jsonl first.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump({"vectorizer": vec, "model": clf}, args.out)
    print(f"\nSaved model to: {args.out}")

if __name__ == "__main__":
    main()
