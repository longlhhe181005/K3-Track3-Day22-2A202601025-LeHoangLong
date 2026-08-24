# Preference Alignment Experiment Report

## 1. Dataset Analysis & Cleaning

### Data Loading Summary
- **Total examples loaded**: `24`
- **Validation issues found**: Line 1 of `data/sample_preferences.jsonl` had unescaped inner double
  quotes around `"self-attention"`, which broke JSON parsing (`json.JSONDecodeError`). All 24 prompts
  were otherwise unique and each `chosen`/`rejected` pair differed.
- **Cleaning steps taken**: Escaped the inner quotes on line 1 (`\"self-attention\"`) so the row
  parses as valid JSON. `load_jsonl` was rewritten to raise a `ValueError` that includes the 1-based
  line number for both JSON decode errors and Pydantic schema errors, to reject duplicate prompts
  (case/whitespace-insensitive) with the line number of the first occurrence, and to warn on crude
  PII markers (e.g. `@`, `ssn:`) found in the text.

### Split Strategy
- **Train/Val Ratio**: `80/20` (`validation_ratio=0.2`, the default in `configs/local.yaml`).
- **Leakage Prevention**: `split_by_prompt` groups examples by a normalized (whitespace-collapsed,
  case-folded) prompt key before splitting, so a given prompt's group can only land entirely in train
  or entirely in validation. Groups are shuffled with a `random.Random(seed)` instance seeded from the
  config (`seed: 42`), so the split is reproducible across runs.

## 2. Implementation: DPO

### Objective Selection
- **Why this method?**: DPO is the simpler of the two objectives to reason about numerically (a
  single log-sigmoid over a policy/reference log-ratio) and matches the config default
  (`training.method: dpo` in `configs/local.yaml`), so it was implemented first. `orpo_loss` is also
  implemented and covered by tests, and the trainer will switch to it if `method: orpo` is set.
- **Key Hyperparameters**:
    - `beta`: `0.1`
    - `lambda_orpo` (if applicable): `0.1`

### Numerical Stability
- **Challenges**: The naive DPO loss `-log(sigmoid(x))` overflows/underflows for large `|x|` because
  `sigmoid` saturates to exactly `0.0` or `1.0` in floating point, and `log(0)` is `-inf`. ORPO's
  odds-ratio term needs `log(1 - p)` from a log-probability `log(p)`, which is undefined for `p >= 1`.
- **Solutions**: `_log_sigmoid` in `losses.py` uses the identity
  `log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)`, implemented via `np.logaddexp(0.0, -x)`,
  which is stable for any finite `x`. For ORPO, `log(1 - exp(logp))` uses `np.log1p(-np.exp(logp))`,
  which is accurate near `logp -> 0` (i.e. `p -> 1`) as long as inputs are genuine negative
  log-probabilities (`logp < 0`), which holds for the mock trainer's proxy scores.

## 3. Evaluation Results

### Metrics
| Metric | Value |
|---|---|
| Pairwise Accuracy | `75%` (`pref-lab evaluate --config configs/local.yaml`, see `outputs/metrics.json`) |
| Final Loss (Mock/Train) | `0.693` (DPO, CPU mock trainer, see `outputs/training_log.json`) |

Pairwise accuracy uses a deterministic, model-free scorer (`deterministic_score` in `evaluate.py`)
that blends prompt/response lexical overlap (70%) with a length prior (30%), since no trained policy
or GPU is available in this lab environment. The mock trainer's loss uses the same scores mapped into
a pseudo-log-probability space (`-(1 - score)`) as a stand-in for real sequence log-probabilities.

### Qualitative Review
- **Prompt**: `What is the difference between bagging and boosting?`
- **Chosen Response**: `Bagging (e.g., Random Forest) trains models in parallel and averages their
  predictions, while boosting (e.g., Gradient Boosting) trains models sequentially, with each model
  correcting the errors of the previous ones.`
- **Rejected Response**: `Bagging is used for classification tasks, while boosting is used for
  regression tasks.`
- **Model Preference**: `Correct` — the chosen response scores higher because it repeats more
  prompt terms ("bagging", "boosting") and is substantially longer/more substantive.

## 4. Discussion & Failure Modes

- **What went well?**: The data pipeline now fails loudly and precisely (line number + reason) on bad
  JSON, schema violations, and duplicate prompts, which made the line-1 quoting bug trivial to find
  and fix. Grouping by prompt before splitting guarantees zero leakage, verified by a dedicated test.
- **Observed Bias**: The deterministic scorer is biased toward longer responses (length term is
  capped but still rewards verbosity) and toward responses that echo prompt vocabulary rather than
  ones that are simply *correct*. On this dataset the two are correlated (the `chosen` answers are
  both longer and more on-topic), so accuracy still lands at 75%, but a response that is short,
  correct, and uses different words than the prompt would be under-scored — this is a known
  limitation of using a lexical proxy instead of a real model's log-probabilities.
- **Safety**: The regression prompts in `docs/regression_prompts.md` are not automatically executable
  against a real model in this CPU-only lab (no trained policy exists). They are documented as a
  manual/future check to run before/after a real training run once GPU access and a base model are
  available.
