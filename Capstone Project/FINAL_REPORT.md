# v0.6.0 Final Report - Offline Dating-Chat Simulator and Safety Evaluation

## Project Goal
Build an offline dating-chat practice simulator with safety-aware evaluation. The system should be reproducible, measurable, and conservative about unsafe escalation while still producing natural conversation.

## System Summary
- Offline chat loop backed by a local GGUF LLM (llama.cpp).
- Persona profiles and profile cards for consistent voice and identity.
- Style planning to guide tone, disclosure, and question-asking.
- Safety gate using a learned classifier with thresholding.
- Trust and phase progression to control escalation behavior.

## What Was Evaluated
The final evaluation focuses on the learned safety classifier using the synthetic validation set. Metrics include confusion matrix and precision/recall/F1/accuracy, with a per-use-case breakdown (UC3 and UC4).

## Final Evaluation Command
```bash
python -m src.eval_final_v0_6
```

Outputs:
- `data/results/final_v0_6_report.json`

## Final Results (v0.6.0)
Source: `data/results/final_v0_6_report.json`

Overall:
- Precision: 0.925
- Recall: 1.000
- F1: 0.961
- Accuracy: 0.970
- Confusion: TP=37, TN=60, FP=3, FN=0

By use case:
- UC3_SUGGEST_DATE: Precision 0.889, Recall 1.000, F1 0.941, Acc 0.958
- UC4_BOUNDARY: Precision 0.955, Recall 1.000, F1 0.977, Acc 0.981

## Reproducible Commands
Chat:
```bash
python -u -m src.chat_v0_5_chatbot \
  --gguf_model models/gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --persona_profile random \
  --threshold 0.45
```

Final eval:
```bash
python -m src.eval_final_v0_6
```

## Data and Artifacts
- Validation dataset: `data/labels_safe_move_synth_validation.jsonl`
- Safety model: `models/safe_violation_clf_embed.joblib`
- Report: `data/results/final_v0_6_report.json`

## Limitations
- Phase detection is heuristic and can misclassify tone.
- Synthetic validation is narrow in domain and tone diversity.
- Safety gate is conservative on borderline language.
- Reported metrics reflect proxy labels derived from rubric scores.

## Freeze Statement
v0.6.0 is FINAL and frozen. No further features are planned.
