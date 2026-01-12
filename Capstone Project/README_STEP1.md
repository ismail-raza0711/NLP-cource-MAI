# Step 1 — Scope and Success Metrics (Capstone NLP)

Note: This document covers the initial Step 1 deliverable only. For current project status and workflows, see `README.md`.

## v0.4 Offline Chatbot (base for v0.5)
Run the terminal chatbot with a local GGUF instruct model (no internet needed once the model is downloaded).

Example:
```bash
python -u -m src.chat_v0_5_chatbot --gguf_model models/gguf/Phi-3-mini-4k-instruct-q4.gguf --persona flirty_adult_ok --threshold 0.45
```

## v0.5.2 Conversation Phases + Personality + Memory
Adds phase tracking, randomized personality profiles, gated erotic escalation, fact-only semantic memory, trust-tiered safety, and identity guards.

Phases: OPENING → RAPPORT → FLIRTING → INTIMATE → EROTIC (with BOUNDARY_REPAIR and COOLDOWN when needed).

Example:
```bash
python -u -m src.chat_v0_5_chatbot \
  --gguf_model models/gguf/Phi-3-mini-4k-instruct-q4.gguf \
  --persona_profile random \
  --threshold 0.45
```

Clear memory:
```bash
python -m src.chat_v0_5_chatbot --clear-memory --memory_id test1
```

Key flags:
- `--persona_profile`: `random` or a preset id (see startup log)
- `--bot_gender`: `female|male|random` (default random)
- `--user_gender`: `female|male|unspecified` (default unspecified)
- `--attraction`: `women|men|any|unspecified` (default unspecified)
- `--memory_id`: session memory file key (default derived from persona + date)
- `--clear-memory`: delete memory file and exit

Memory behavior: extracts simple user facts (likes/favorites/job) and skips sensitive data (addresses/phones).
Blocking behavior: repeated escalation or low engagement can trigger a polite block and end the session.
Trust behavior: trust level grows with respectful, mutual signals and governs when suggestive/explicit content is allowed.
Identity behavior: bot name/gender/pronouns are locked; implausible claims are deflected.

## Purpose
This repository defines a platform-agnostic dating-chat practice simulator dataset and evaluation rubric for an NLP capstone project.

## In Scope
- 4 use cases: cold open, keep-going, suggest-date, boundary moment
- Unit: user message scored in context of persona profile + chat history
- Rubric: 6 dimensions (0–2 each), total 0–12, normalized to OCQ in [0,1]
- Data stored in JSON/JSONL for reproducibility

## Out of Scope / Constraints
- No app integration or scraping
- No coaching for harassment, coercion, manipulation, or explicit sexual content
- Store only synthetic IDs; no personal identifiers

## Rubric Dimensions
ENG (Engagement), CTX (Context use), TONE (Tone/politeness),
CLAR (Clarity), SAFE (Boundary/safety), MOVE (Forward motion)

## Metrics
- OCQ = (ENG+CTX+TONE+CLAR+SAFE+MOVE)/12
- ENG_score = ENG/2
- CTX_score = CTX/2
- Safety violation flag = 1 if SAFE==0 else 0
- Dataset-level: mean/median OCQ, violation rate, by-use-case breakdown

## Files
- data/personas.json: persona profiles
- data/contexts.jsonl: context situations by use case
- data/samples_unlabeled.jsonl: user message samples to annotate
- data/labels_template.jsonl: annotation template aligned by sample_id
- data/labels_gold_example.jsonl: a small example of completed labels

## Reproducible generation
`./src/make_step1_data.py` writes the Step 1 dataset files.
`./src/score_report.py` computes metrics from a labeled file.
