"""Group-relative policy optimization, reduced to the only part that is actually GRPO.

An RL step on an LLM and an RL step on a 100k-parameter MNIST classifier differ in
scale, not in structure:

    1. sample K actions per prompt from the current policy
    2. score each action with a reward function
    3. turn rewards into advantages *relative to the group of K*
    4. take a policy-gradient step weighted by those advantages

Step 3 is GRPO. It is the function below, and it is 6 lines long. Everything that
makes LLM RL expensive lives in steps 1 and 2, not here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "group_advantages",
    "reinforce_advantages",
    "pg_loss",
    "group_reward_std",
]


def _as_groups(rewards: np.ndarray) -> np.ndarray:
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(
            f"rewards must be 2-D (n_prompts, n_samples_per_prompt); got shape {r.shape}"
        )
    if r.shape[1] < 1:
        raise ValueError("n_samples_per_prompt must be >= 1")
    return r


def group_advantages(
    rewards: np.ndarray,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """GRPO advantages: center each prompt's rewards on that prompt's own mean.

    Args:
        rewards: shape ``(n_prompts, n_samples_per_prompt)``.
        normalize: also divide by the group's standard deviation. This is the
            "normalized" GRPO variant; it makes the step size invariant to how
            spread out a prompt's rewards happen to be.
        eps: floor on the divisor.

    Returns:
        Advantages, same shape as ``rewards``. Each row sums to ~0.

    Note:
        A group whose rewards are all identical yields all-zero advantages, i.e.
        *no gradient signal at all*. Prompts that are always solved and prompts
        that are never solved both teach the policy nothing. This is the single
        most important practical fact about GRPO: it is why you filter your
        dataset for prompts of middling difficulty, and why ``n_samples_per_prompt``
        of 1 cannot work.
    """
    r = _as_groups(rewards)
    adv = r - r.mean(axis=1, keepdims=True)
    if normalize:
        adv = adv / (r.std(axis=1, keepdims=True) + eps)
    return adv


def reinforce_advantages(rewards: np.ndarray) -> np.ndarray:
    """REINFORCE-with-baseline advantages: center on the *batch* mean.

    The contrast with :func:`group_advantages` is the point of this file. Here the
    baseline is shared across prompts, so an easy prompt's high reward is credited
    to the action taken rather than to the prompt being easy.
    """
    r = _as_groups(rewards)
    return r - r.mean()


def group_reward_std(rewards: np.ndarray) -> np.ndarray:
    """Per-prompt reward std — the diagnostic for "is this prompt teaching anything".

    Rows at 0.0 are degenerate groups (all samples scored identically).
    """
    return _as_groups(rewards).std(axis=1)


def pg_loss(action_log_probs: np.ndarray, advantages: np.ndarray) -> float:
    """The policy-gradient objective: ``-mean(advantage * log pi(action))``.

    No importance ratio and no clipping, because this is strictly on-policy: the
    samples were drawn from the same weights we are about to update. PPO/GRPO
    clipping exists to permit multiple optimizer epochs over one batch of
    rollouts; with a single epoch the ratio is identically 1 and clipping is inert.
    """
    lp = np.asarray(action_log_probs, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    if lp.shape != adv.shape:
        raise ValueError(f"shape mismatch: {lp.shape} vs {adv.shape}")
    return float(-(adv * lp).mean())
