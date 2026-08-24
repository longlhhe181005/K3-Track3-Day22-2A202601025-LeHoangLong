from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import load_jsonl
from .evaluate import deterministic_score
from .losses import dpo_loss, orpo_loss


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    data_path: str = "data/sample_preferences.jsonl"
    output_dir: str = "outputs"

class PreferenceTrainer:
    """Interface for DPO/ORPO training implementations."""
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> None:
        """Train the policy.

        CPU-only mock trainer: there is no GPU/model backend in this lab, so we
        approximate "training" by scoring chosen/rejected responses with a
        deterministic scorer, feeding those as proxy log-probabilities into the
        configured loss, and logging the loss per batch. Checkpoints are not
        produced (no real weights exist), but a training log is written to
        `output_dir/training_log.json`.
        """
        examples = load_jsonl(self.config.data_path)
        if not examples:
            raise ValueError(f"No training examples found at {self.config.data_path}")

        chosen_scores = np.array([deterministic_score(ex.prompt, ex.chosen) for ex in examples])
        rejected_scores = np.array([deterministic_score(ex.prompt, ex.rejected) for ex in examples])
        # Map bounded scores into negative-log-probability space (higher score -> less negative).
        chosen_logps = -(1.0 - chosen_scores)
        rejected_logps = -(1.0 - rejected_scores)

        batch_losses: list[float] = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(examples), batch_size):
            end = start + batch_size
            c_logps = chosen_logps[start:end]
            r_logps = rejected_logps[start:end]
            if self.config.method == "orpo":
                sft_nll = -c_logps
                loss = orpo_loss(sft_nll, c_logps, r_logps, self.config.lambda_orpo)
            else:
                # DPO with a frozen reference at zero log-ratio (uniform reference proxy).
                ref_logps = np.zeros_like(c_logps)
                loss = dpo_loss(c_logps, r_logps, ref_logps, ref_logps, self.config.beta)
            batch_losses.append(loss)

        log = {
            "method": self.config.method,
            "num_examples": len(examples),
            "num_batches": len(batch_losses),
            "batch_losses": batch_losses,
            "final_loss": batch_losses[-1] if batch_losses else None,
        }
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
