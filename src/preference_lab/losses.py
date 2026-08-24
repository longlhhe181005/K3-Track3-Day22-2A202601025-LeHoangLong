from __future__ import annotations

import numpy as np


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(sigmoid(x)) = -softplus(-x) = -log1p(exp(-x))."""
    result: np.ndarray = -np.logaddexp(0.0, -x)
    return result

def dpo_loss(policy_chosen_logps: np.ndarray, policy_rejected_logps: np.ndarray, ref_chosen_logps: np.ndarray, ref_rejected_logps: np.ndarray, beta: float) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    Compares the policy log-ratio (chosen vs. rejected) against the same ratio
    under the reference model, scaled by beta, then applies a stable log-sigmoid.
    """
    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratio - ref_logratio)
    loss = -_log_sigmoid(logits)
    return float(np.mean(loss))

def orpo_loss(sft_nll: np.ndarray, chosen_logps: np.ndarray, rejected_logps: np.ndarray, lambda_orpo: float) -> float:
    """Compute a simplified ORPO-style objective.

    Combines the SFT negative log-likelihood on the chosen response with an
    odds-ratio penalty that pushes the chosen log-odds above the rejected one.
    """
    log_odds_chosen = chosen_logps - np.log1p(-np.exp(chosen_logps))
    log_odds_rejected = rejected_logps - np.log1p(-np.exp(rejected_logps))
    odds_ratio_penalty = -_log_sigmoid(log_odds_chosen - log_odds_rejected)
    loss = sft_nll + lambda_orpo * odds_ratio_penalty
    return float(np.mean(loss))
