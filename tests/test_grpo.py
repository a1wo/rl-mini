"""The advantage math, including the failure mode that wastes the most GPU hours."""

import numpy as np
import pytest

from rl_mini.grpo import (
    group_advantages,
    group_reward_std,
    pg_loss,
    reinforce_advantages,
)


def test_group_advantages_center_each_group():
    rewards = np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    adv = group_advantages(rewards, normalize=False)
    assert np.allclose(adv.sum(axis=1), 0.0)
    assert np.allclose(adv, rewards - rewards.mean(axis=1, keepdims=True))


def test_normalized_group_advantages_are_unit_scale():
    rewards = np.array([[0.0, 1.0, 1.0, 0.0]])
    adv = group_advantages(rewards, normalize=True)
    assert np.allclose(adv.std(), 1.0, atol=1e-6)


def test_all_correct_group_gives_no_gradient_signal():
    """A prompt the model always solves teaches it nothing. Same for always-fails.

    This is why you filter an RL dataset for middling difficulty, and why the
    coding-agent tutorial in modal-dojo keeps only tasks with mixed outcomes.
    """
    solved = np.ones((3, 8))
    failed = np.zeros((3, 8))
    assert np.allclose(group_advantages(solved), 0.0)
    assert np.allclose(group_advantages(failed), 0.0)
    assert np.allclose(group_reward_std(solved), 0.0)


def test_grpo_differs_from_reinforce_on_an_easy_prompt():
    # Prompt 0 is easy (mostly solved), prompt 1 is hard (mostly failed).
    rewards = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    grpo = group_advantages(rewards, normalize=False)
    reinforce = reinforce_advantages(rewards)
    # Under GRPO the lone failure on the easy prompt is punished just as hard as
    # the lone success on the hard prompt is rewarded: symmetric, prompt-relative.
    assert grpo[0, 3] == pytest.approx(-grpo[1, 3])
    # Under a shared baseline the easy prompt's successes still look good, so the
    # policy is pushed toward what it already does well.
    assert reinforce[0, 0] > 0
    assert not np.allclose(grpo, reinforce)


def test_group_advantages_rejects_bad_shapes():
    with pytest.raises(ValueError):
        group_advantages(np.array([1.0, 0.0]))


def test_pg_loss_sign():
    log_probs = np.log(np.array([[0.5, 0.5]]))
    # Positive advantage on an action => loss decreases when that action gets
    # more likely, i.e. the loss is negative-advantage-weighted log prob.
    assert pg_loss(log_probs, np.array([[1.0, 0.0]])) == pytest.approx(-0.5 * np.log(0.5))
    assert pg_loss(log_probs, np.array([[0.0, 0.0]])) == pytest.approx(0.0)
