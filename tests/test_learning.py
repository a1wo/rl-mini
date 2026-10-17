"""End-to-end: does the loop actually learn? Fast, deterministic, CPU-only.

These are the tests that would catch a silently-broken RL implementation. A
gradient check (tests/test_policy_grad.py) proves the arithmetic is right; only a
learning test proves the *loop* is right.
"""

import numpy as np
import pytest

from rl_mini.env import BernoulliBandit, ClassificationEnv
from rl_mini.grpo import group_advantages, reinforce_advantages
from rl_mini.policy import Adam, MLPPolicy


def _train_bandit(estimator: str, *, seed: int, steps: int = 300, n_samples: int = 16):
    """One context, 4 arms, arm 2 is best. The optimal policy is known in closed form."""
    bandit = BernoulliBandit([0.1, 0.2, 0.9, 0.3], seed=seed)
    policy = MLPPolicy(n_features=1, n_actions=4, hidden=8, seed=seed)
    optimizer = Adam(policy.param_dict(), lr=0.05)
    rng = np.random.default_rng(seed)
    X = np.ones((1, 1))

    for _ in range(steps):
        actions = policy.sample_actions(X, n_samples, rng)
        rewards = bandit.reward(actions)
        adv = (
            group_advantages(rewards)
            if estimator == "grpo"
            else reinforce_advantages(rewards)
        )
        _, grads = policy.pg_loss_and_grads(X, actions, adv, entropy_coef=0.001)
        optimizer.step(grads)
    return policy, bandit


@pytest.mark.parametrize("estimator", ["grpo", "reinforce"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_policy_gradient_finds_the_best_arm(estimator, seed):
    policy, bandit = _train_bandit(estimator, seed=seed)
    probs = policy.probs(np.ones((1, 1)))[0]
    assert int(np.argmax(probs)) == bandit.best_action
    assert probs[bandit.best_action] > 0.85


def _synthetic_classification(seed: int = 0, n_classes: int = 8, n_features: int = 16):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, n_features)) * 3.0
    labels = rng.integers(0, n_classes, size=512)
    X = centers[labels] + rng.standard_normal((512, n_features)) * 0.3
    return X, labels, n_classes


def _train_classifier(final_layer_scale: float, steps: int = 150) -> tuple[float, set[int]]:
    X, labels, n_classes = _synthetic_classification()
    policy = MLPPolicy(
        X.shape[1], n_classes, hidden=32, seed=0, final_layer_scale=final_layer_scale
    )
    optimizer = Adam(policy.param_dict(), lr=0.02)
    env = ClassificationEnv(n_actions=n_classes)
    rng = np.random.default_rng(1)

    for _ in range(steps):
        index = rng.integers(0, X.shape[0], size=64)
        batch_X, batch_y = X[index], labels[index]
        actions = policy.sample_actions(batch_X, 8, rng)
        rewards = env.reward(actions, batch_y)
        adv = group_advantages(rewards)
        _, grads = policy.pg_loss_and_grads(batch_X, actions, adv, entropy_coef=0.01)
        optimizer.step(grads)

    predictions = policy.greedy_actions(X)
    accuracy = float((predictions == labels).mean())
    never_predicted = set(range(n_classes)) - set(predictions.tolist())
    return accuracy, never_predicted


def test_grpo_learns_a_classifier_from_reward_alone():
    """The learner is told only whether each guess was right, never the label."""
    accuracy, never_predicted = _train_classifier(final_layer_scale=0.01)
    assert accuracy > 0.99, f"RL failed to learn: {accuracy:.3f}"
    assert not never_predicted


def test_peaked_initialization_starves_exploration():
    """A controlled demonstration of the most common silent RL failure.

    Identical data, network, optimizer, reward and number of steps. The only
    change is the scale of the output layer at initialization, i.e. how
    deterministic the *starting policy* is.

    With a standard He-scaled output layer the initial policy is sharply peaked,
    some actions are essentially never sampled, and a policy gradient cannot learn
    about an action it never takes -- so those classes stay unpredicted forever
    and accuracy plateaus at a suspiciously round fraction (here 7/8).

    The supervised control in test_policy_grad.py passes either way: this failure
    is specific to learning from rewards.
    """
    peaked_accuracy, peaked_missing = _train_classifier(final_layer_scale=1.0)
    uniform_accuracy, uniform_missing = _train_classifier(final_layer_scale=0.01)

    assert peaked_missing, "expected the peaked init to abandon at least one action"
    assert peaked_accuracy == pytest.approx(7 / 8, abs=0.02)
    assert uniform_accuracy > peaked_accuracy + 0.1
    assert not uniform_missing


def test_near_uniform_init_is_the_default():
    policy = MLPPolicy(n_features=16, n_actions=8, hidden=32, seed=0)
    probs = policy.probs(np.random.default_rng(0).standard_normal((64, 16)))
    assert np.allclose(probs.mean(axis=0), 1 / 8, atol=0.02)


def test_degenerate_groups_produce_a_zero_gradient():
    """If every sample in a group scores the same, the update is exactly zero."""
    policy = MLPPolicy(4, 3, hidden=5, seed=0)
    X = np.random.default_rng(0).standard_normal((2, 4))
    actions = np.array([[0, 1, 2, 0], [1, 1, 0, 2]])
    rewards = np.ones((2, 4))  # every sample solved
    _, grads = policy.pg_loss_and_grads(X, actions, group_advantages(rewards))
    for grad in grads.values():
        assert np.allclose(grad, 0.0)
