# src/safety_embed.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class SafetyScore:
    p_move: float
    label: str
    threshold: float

    def as_dict(self) -> Dict[str, Any]:
        return {"p_move": float(self.p_move), "label": self.label, "threshold": float(self.threshold)}


class SafetyEmbedScorer:
    """
    Loads the v0.3 embedding+logreg artifact and scores text with p(MOVE).

    Expected artifact keys (as produced in your project):
      - sentence_transformer: str
      - logreg: sklearn LogisticRegression
      - normalize_embeddings: bool (optional)
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.artifact = joblib.load(model_path)
        self.embed_name = self.artifact["sentence_transformer"]
        self.clf = self.artifact["logreg"]
        self.normalize = bool(self.artifact.get("normalize_embeddings", True))
        self.embedder = SentenceTransformer(self.embed_name)

    def predict_proba_move(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return 0.0

        X = self.embedder.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        p_move = float(self.clf.predict_proba(X)[0, 1])
        # numerical safety
        if np.isnan(p_move) or np.isinf(p_move):
            return 0.0
        return max(0.0, min(1.0, p_move))

    def score(self, text: str, threshold: float = 0.45) -> SafetyScore:
        p = self.predict_proba_move(text)
        label = "MOVE" if p >= threshold else "SAFE"
        return SafetyScore(p_move=p, label=label, threshold=threshold)
