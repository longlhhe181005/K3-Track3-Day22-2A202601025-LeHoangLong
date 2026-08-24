# Data Card

- **Dataset name**: `sample_preferences` (preference-alignment-lab-starter)
- **Source**: Hand-authored/curated preference pairs covering conceptual machine learning
  questions (transformers, backprop, GANs, regularization, evaluation metrics, etc.), stored at
  `data/sample_preferences.jsonl`.
- **License/permission**: Internal lab material; no third-party copyrighted text included. Safe for
  use within this course.
- **Schema**: One JSON object per line: `prompt: str`, `chosen: str`, `rejected: str`,
  `metadata: {domain: str, rubric: str}`. Enforced by `preference_lab.schemas.PreferenceExample`
  (min length 1, whitespace-stripped, `chosen != rejected` after case/whitespace normalization).
- **Labeling rubric**: `accuracy` — the `chosen` response must be a factually correct, complete
  explanation of the concept asked; the `rejected` response is a plausible-sounding but factually
  incorrect or off-target answer to the same prompt.
- **Known biases**: All 24 examples share `domain: education` / `rubric: accuracy`, and `chosen`
  responses are consistently longer and more terminology-dense than `rejected` ones. A model or
  scorer trained/evaluated on this data alone will conflate "longer and more technical" with
  "better," and will not have seen examples testing conciseness, tone, safety, or refusal behavior.
- **Safety/PII checks**: `load_jsonl` scans for crude PII markers (`@`, `ssn:`) and logs a warning;
  no PII was found in this dataset (all content is abstract ML explanations, no personal data).
  Regression prompts for safety-adjacent behavior (medical advice, admitting uncertainty, etc.) are
  tracked separately in `docs/regression_prompts.md` and are not part of this training data.
- **Train/validation/test split method**: `split_by_prompt` (in `preference_lab/data.py`) groups
  rows by normalized prompt text, shuffles the groups deterministically with a configurable seed
  (`seed: 42` in `configs/local.yaml`), then cuts ~20% of groups into validation. This guarantees no
  prompt appears in both splits. No separate held-out test set is defined for this lab; the
  validation split doubles as the evaluation set for `pref-lab evaluate`.
