import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_prefers_matching_policy_ratio() -> None:
    # Policy log-ratio matches reference exactly -> logits are 0 -> loss = log(2).
    loss = dpo_loss(
        np.array([-0.5]), np.array([-1.5]), np.array([-0.5]), np.array([-1.5]), beta=0.1
    )
    assert loss == pytest.approx(np.log(2), rel=1e-6)

def test_dpo_loss_rewards_larger_chosen_margin() -> None:
    # Policy separates chosen/rejected more than the reference -> lower loss.
    high_margin = dpo_loss(
        np.array([-0.1]), np.array([-2.0]), np.array([-0.6]), np.array([-1.0]), beta=0.1
    )
    low_margin = dpo_loss(
        np.array([-0.6]), np.array([-1.0]), np.array([-0.6]), np.array([-1.0]), beta=0.1
    )
    assert high_margin < low_margin

def test_orpo_loss_is_finite_and_positive() -> None:
    loss = orpo_loss(np.array([1.0]), np.array([-0.5]), np.array([-1.5]), lambda_orpo=0.1)
    assert np.isfinite(loss)
    assert loss > 0

def test_orpo_loss_penalizes_worse_odds_ratio() -> None:
    good = orpo_loss(np.array([1.0]), np.array([-0.1]), np.array([-2.0]), lambda_orpo=0.5)
    bad = orpo_loss(np.array([1.0]), np.array([-1.0]), np.array([-0.1]), lambda_orpo=0.5)
    assert good < bad
