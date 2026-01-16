# Changelog

All notable changes to this project are documented in this file.

This project follows a versioned, experimental development process suitable for an NLP research capstone. Each version represents a frozen, reproducible system state used for evaluation and comparison.

---

## [v0.6.0] - Real Profiles + Style Planner (FINAL)

Frozen final release (no further features).

### Added
- Conversation phase tracking with conservative transitions and debug output.
- Randomized personality profiles per session.
- Gated erotic escalation with soft deflection templates.
- Fact-only semantic memory persisted under `data/memory/`.
- Context-aware safety repair responses.
- Bot gender selection and neutral attraction handling.
- SAFE expansion synthetic file for benign intimacy-adjacent language.
- Merge script and merged training set for SAFE expansion data.
- Session blocking for repeated escalation or low engagement.
- Trust-level safety tiers with consent logging.
- Identity lock and reality guard for consistent, plausible replies.
- Real profile cards (bio + photos) and deterministic /profile /pics /name replies.
- Personality-driven response planner with style debug output.
- Final evaluation runner `src/eval_final_v0_6.py` with output at `data/results/final_v0_6_report.json`.
- Final project README with reproducible commands.

### Changed
- Chat entrypoint updated to `python -m src.chat_v0_5_chatbot`.
- System prompt now injects phase/personality/memory context for the LLM.
 - Safety repair now acknowledges user content before redirecting.

### Known Limitations
- Phase/intent detection is heuristic and can misclassify tone.
- Safety gate may still be conservative on borderline messages.

### Purpose
Adds realism and continuity to the offline chatbot while keeping the v0.3 safety gate as a guardrail.

---

## [v0.2.0] — Learned Safety Intervention

### Added
- Weak-supervision pipeline for generating SAFE/MOVE labels.
- Automatic materialization of supervised training data:
  - `labels_safe_move_gold_template.jsonl` → `labels_safe_move_gold.jsonl`
- TF-IDF + Logistic Regression classifier for predicting safety violations.
- Interactive chat loop with:
  - Probabilistic safety risk estimation.
  - Threshold-based intervention.
  - Scenario-aligned rewrite suggestions.
- Batch evaluation script using learned safety model.
- Episode logging for interaction analysis.

### Changed
- Expanded dataset with targeted UC3 (Suggest Date) and UC4 (Boundary) scenarios.
- Restricted v0.2 interaction to escalation- and boundary-relevant use cases.
- Transitioned from heuristic-only safety checks to learned, probabilistic intervention.

### Known Limitations
- Safety classifier inherits blind spots from weak-supervision rules.
- Implicit sexual language not always detected.
- Partner responses remain rule-based and simplistic.

### Purpose
Demonstrates learned safety-aware intervention in conversational interaction, enabling quantitative comparison against a heuristic baseline.

---

## [v0.1.0] — Heuristic Baseline with Batch Evaluation

### Added
- Rule-based conversational quality scoring (ENG, CTX, TONE, CLAR, SAFE, MOVE).
- Overall Conversation Quality (OCQ) metric.
- Batch evaluation pipeline over fixed dataset.
- Per–use-case performance reporting.
- Safety violation rate analysis.

### Purpose
Establishes a non-trivial heuristic baseline and evaluation framework to support future learned models.

---

## [v0.0.0] — Initial Baseline System

### Added
- Interactive, rule-based chatbot for Tinder-style conversation practice.
- Persona- and context-driven interaction setup.
- Deterministic scoring heuristics for engagement and safety.
- Project structure, documentation, and Git versioning.

### Purpose
Provides a runnable baseline system and foundational project scaffolding.

---

## Versioning Notes
- Versions are tagged in Git and represent frozen experimental checkpoints.
- Later versions are evaluated against earlier ones using identical datasets and metrics.
- The project intentionally progresses from deterministic heuristics to learned models.
