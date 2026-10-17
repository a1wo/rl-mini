"""Environments: the thing that turns an action into a reward.

modal-dojo's docs put it this way: "An environment specifies how the model acts and
how its responses are rewarded." These are the two smallest environments that still
exercise a real policy-gradient loop.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ClassificationEnv", "BernoulliBandit"]


class ClassificationEnv:
    """Classification as a one-step contextual bandit.

    The policy sees an image and emits *one* digit. It is told only whether that
    digit was right — never what the right answer was. That is the whole difference
    between this and supervised learning, and it is the same information structure
    an LLM gets from a verifiable reward.

    Args:
        n_actions: size of the action space (10 for MNIST).
        partial_credit: reward for a wrong action. Keep at 0.0 for the honest
            sparse-reward setting; raise it to soften the signal.
    """

    def __init__(self, n_actions: int = 10, partial_credit: float = 0.0):
        self.n_actions = n_actions
        self.partial_credit = float(partial_credit)

    def reward(self, actions: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Score actions.

        Args:
            actions: shape ``(n_prompts, n_samples_per_prompt)``.
            labels: shape ``(n_prompts,)``.

        Returns:
            Rewards, shape ``(n_prompts, n_samples_per_prompt)``.
        """
        actions = np.asarray(actions)
        labels = np.asarray(labels)
        if actions.ndim != 2:
            raise ValueError(f"actions must be 2-D; got {actions.shape}")
        if labels.shape[0] != actions.shape[0]:
            raise ValueError(
                f"labels ({labels.shape[0]}) must match n_prompts ({actions.shape[0]})"
            )
        hit = actions == labels[:, None]
        return np.where(hit, 1.0, self.partial_credit)


class BernoulliBandit:
    """A k-armed bandit with fixed per-arm success probabilities.

    Used in the tests as a ground-truth check that the policy gradient climbs: the
    optimal policy is known in closed form (always pull ``argmax(probs)``).
    """

    def __init__(self, probs: np.ndarray, seed: int = 0):
        self.probs = np.asarray(probs, dtype=np.float64)
        if self.probs.ndim != 1 or np.any(self.probs < 0) or np.any(self.probs > 1):
            raise ValueError("probs must be a 1-D array of probabilities")
        self.rng = np.random.default_rng(seed)

    @property
    def n_actions(self) -> int:
        return int(self.probs.shape[0])

    @property
    def best_action(self) -> int:
        return int(np.argmax(self.probs))

    def reward(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions)
        draws = self.rng.random(actions.shape)
        return (draws < self.probs[actions]).astype(np.float64)
