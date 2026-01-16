# v0.6.0 FINAL - Dating-Chat Practice Simulator (NLP Capstone)

This project is an offline dating-chat practice simulator plus a safety-aware evaluation pipeline. It progresses from heuristic scoring to a learned safety gate, then adds realism with phase tracking, trust tiers, and persona/profile consistency. v0.6.0 is FINAL and frozen (no further feature work).

## What Was Tested
- Heuristic scoring baselines (ENG/CTX/TONE/CLAR/SAFE/MOVE).
- Learned safety intervention (TF-IDF + logistic regression).
- Phase/trust gating realism layered on the offline chat simulator.

## System Summary
- Offline simulator driven by a local GGUF LLM (llama.cpp).
- Persona profiles, profile cards, and style planning for consistent replies.
- Safety-aware gating (risk threshold + boundary repair).
- Trust/phase progression controls erotic escalation.
- Rubric dimensions: ENG, CTX, TONE, CLAR, SAFE, MOVE.

## Setup
- Python: 3.10+ (tested on 3.10/3.11).
- Install deps:
  - `pip install -r requirements.txt`
- You need a local GGUF instruct model (example path below).

## Run Chat (v0.6.0)
```bash
python -u -m src.chat_v0_5_chatbot \
  --gguf_model models/gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --persona_profile random \
  --threshold 0.45
```

## Reproduce Final Evaluation
Single command from repo root:
```bash
python -m src.eval_final_v0_6
```

Outputs:
- `data/results/final_v0_6_report.json`

## Notes on Reproducibility
- The final evaluation is deterministic given the same model artifact and dataset.
- If you run with GPU embeddings, tiny floating-point differences may occur.
- Dependencies are listed but not pinned; pin them for strict reproduction.

## Limitations / Known Issues
- Phase detection is heuristic and can misclassify tone.
- Synthetic validation data is limited in scope and tone diversity.
- Safety gating is conservative on borderline language.
- The safety gate is a proxy trained on MOVE labels from synthetic rubric scores.

## Documentation
- `FINAL_REPORT.md`: final narrative report and results summary.
- `SLIDES_OUTLINE.md`: presentation outline.
- `README_STEP1.md`: initial scope and rubric definitions.
- `data/DATASET.md`: dataset file schemas.
- `data/EVAL.md`: evaluation scripts and past reports.
- `CHANGELOG.md`: version history (v0.6.0 FINAL).
