"""Finite-difference checks: the hand-written gradients are verified, not asserted."""

import numpy as np
import pytest

from rl_mini.grpo import group_advantages
from rl_mini.policy import MLPPolicy


def _numeric_grad(fn, param, eps=1e-6):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + eps
        plus = fn()
        param[idx] = original - eps
        minus = fn()
        param[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


@pytest.mark.parametrize("entropy_coef", [0.0, 0.05])
def test_pg_gradients_match_finite_differences(entropy_coef):
    rng = np.random.default_rng(0)
    policy = MLPPolicy(n_features=6, n_actions=4, hidden=5, seed=1)
    X = rng.standard_normal((3, 6))
    actions = rng.integers(0, 4, size=(3, 5))
    advantages = group_advantages(rng.random((3, 5)))

    loss, grads = policy.pg_loss_and_grads(X, actions, advantages, entropy_coef)
    assert np.isfinite(loss)

    for name, param in policy.param_dict().items():
        numeric = _numeric_grad(
            lambda: policy.pg_loss_and_grads(X, actions, advantages, entropy_coef)[0],
            param,
        )
        assert np.allclose(grads[name], numeric, atol=1e-6), f"gradient mismatch for {name}"


def test_cross_entropy_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    policy = MLPPolicy(n_features=6, n_actions=4, hidden=5, seed=3)
    X = rng.standard_normal((4, 6))
    labels = rng.integers(0, 4, size=4)

    _, grads = policy.ce_loss_and_grads(X, labels)
    for name, param in policy.param_dict().items():
        numeric = _numeric_grad(lambda: policy.ce_loss_and_grads(X, labels)[0], param)
        assert np.allclose(grads[name], numeric, atol=1e-6), f"gradient mismatch for {name}"


def test_sampling_follows_the_policy_distribution():
    policy = MLPPolicy(n_features=2, n_actions=3, hidden=4, seed=4)
    X = np.zeros((1, 2))
    rng = np.random.default_rng(5)
    actions = policy.sample_actions(X, 20000, rng)
    empirical = np.bincount(actions.ravel(), minlength=3) / actions.size
    assert np.allclose(empirical, policy.probs(X)[0], atol=0.02)
    assert actions.max() < 3 and actions.min() >= 0


def test_log_probs_are_normalized():
    policy = MLPPolicy(n_features=3, n_actions=5, hidden=4, seed=6)
    X = np.random.default_rng(7).standard_normal((8, 3))
    assert np.allclose(np.exp(policy.log_probs(X)).sum(axis=1), 1.0)
