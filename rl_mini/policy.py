"""A small categorical policy (2-layer MLP) with hand-written gradients.

Deliberately numpy-only: no torch, no CUDA, no framework. The gradient of the
policy-gradient objective is ~10 lines, and writing it out is the fastest way to
see that "RL on an LLM" and "RL on a tiny classifier" are the same update rule.
``tests/test_policy_grad.py`` checks every analytic gradient here against finite
differences, so the arithmetic is verified rather than asserted.
"""

from __future__ import annotations

import numpy as np

__all__ = ["MLPPolicy", "Adam"]


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


class MLPPolicy:
    """``features -> hidden (ReLU) -> logits`` over a discrete action space."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden: int = 128,
        seed: int = 0,
        final_layer_scale: float = 0.01,
    ):
        """
        Args:
            final_layer_scale: shrink factor on the output layer at init.

                This is not a cosmetic default. With standard He initialization the
                initial logits are large, so the initial *policy* is nearly
                deterministic -- measured on an 8-class problem, a He-initialized
                policy here put 59% of its mass on one action and 0.1% on another.
                A policy gradient can only learn about actions it actually samples,
                so the starved actions are never tried, never rewarded, and the run
                plateaus with entire classes that are never predicted. Supervised
                training on the same network does not care, because it is handed the
                label rather than having to discover it.

                Scaling the output layer to ~0 makes the initial policy near-uniform,
                which is maximum exploration. Standard practice in RL
                implementations, and the single highest-leverage line in this file.
                Set it to 1.0 to reproduce the failure (see tests/test_learning.py).
        """
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.n_actions = n_actions
        # He initialization; float64 throughout so finite-difference tests are clean.
        self.W1 = rng.standard_normal((n_features, hidden)) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(hidden)
        self.W2 = (
            rng.standard_normal((hidden, n_actions))
            * np.sqrt(2.0 / hidden)
            * final_layer_scale
        )
        self.b2 = np.zeros(n_actions)

    # ---------------------------------------------------------------- forward

    def param_dict(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(hidden_activations, logits)``."""
        h = np.maximum(X @ self.W1 + self.b1, 0.0)
        return h, h @ self.W2 + self.b2

    def log_probs(self, X: np.ndarray) -> np.ndarray:
        """Log action probabilities, shape ``(n_prompts, n_actions)``."""
        return _log_softmax(self.forward(X)[1])

    def probs(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.log_probs(X))

    def greedy_actions(self, X: np.ndarray) -> np.ndarray:
        """Argmax action — used for evaluation, never for rollouts."""
        return np.argmax(self.forward(X)[1], axis=1)

    def sample_actions(
        self,
        X: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``n_samples`` actions per row, shape ``(n_prompts, n_samples)``.

        This is the rollout. On an LLM it is an SGLang server generating K
        completions per prompt; here it is one categorical draw.
        """
        cumulative = self.probs(X).cumsum(axis=1)
        # Guard against the final cumulative sum landing just under 1.0.
        cumulative[:, -1] = 1.0
        u = rng.random((X.shape[0], n_samples))
        return (u[:, :, None] > cumulative[:, None, :]).sum(axis=2)

    # ------------------------------------------------------------- objectives

    def _backprop_logits(
        self, X: np.ndarray, h: np.ndarray, dlogits: np.ndarray
    ) -> dict[str, np.ndarray]:
        grads = {
            "W2": h.T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
        dz1 = (dlogits @ self.W2.T) * (h > 0)
        grads["W1"] = X.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return grads

    def pg_loss_and_grads(
        self,
        X: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        entropy_coef: float = 0.0,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Policy-gradient loss and its gradients.

        loss = -mean_over_samples(advantage * log pi(action|x))
               - entropy_coef * mean_over_prompts(H[pi(.|x)])

        The entropy bonus is what stops a policy that has found one decent action
        from collapsing onto it before it has tried the others.

        Args:
            X: features, shape ``(n_prompts, n_features)``.
            actions: shape ``(n_prompts, n_samples_per_prompt)``.
            advantages: shape ``(n_prompts, n_samples_per_prompt)``.
            entropy_coef: weight on the entropy bonus.

        Returns:
            ``(loss, grads)`` where ``grads`` is keyed like :meth:`param_dict`.
        """
        actions = np.asarray(actions)
        advantages = np.asarray(advantages, dtype=np.float64)
        if actions.shape != advantages.shape:
            raise ValueError(
                f"actions {actions.shape} and advantages {advantages.shape} must match"
            )
        n_prompts, n_samples = actions.shape
        if X.shape[0] != n_prompts:
            raise ValueError(f"X has {X.shape[0]} rows, actions have {n_prompts}")

        h, logits = self.forward(X)
        logp = _log_softmax(logits)
        p = np.exp(logp)
        n_total = n_prompts * n_samples

        chosen_logp = np.take_along_axis(logp, actions, axis=1)
        loss = float(-(advantages * chosen_logp).sum() / n_total)

        # d/dlogits_j of -A*log pi(a) is  A*(p_j - 1{j==a}), summed over samples.
        scattered = np.zeros_like(p)
        rows = np.repeat(np.arange(n_prompts), n_samples)
        np.add.at(scattered, (rows, actions.ravel()), advantages.ravel())
        dlogits = (p * advantages.sum(axis=1, keepdims=True) - scattered) / n_total

        if entropy_coef:
            entropy = -(p * logp).sum(axis=1)
            loss -= entropy_coef * float(entropy.mean())
            # dH/dlogits_j = -p_j * (log p_j + H)
            dentropy = -p * (logp + entropy[:, None])
            dlogits -= entropy_coef * dentropy / n_prompts

        return loss, self._backprop_logits(X, h, dlogits)

    def ce_loss_and_grads(
        self, X: np.ndarray, labels: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Supervised cross-entropy, for the reference ceiling.

        Same network, same optimizer, same data — the only change is that the
        learner is handed the correct label instead of a scalar reward. The gap
        between this and the RL run is the price of learning from a reward.
        """
        labels = np.asarray(labels)
        h, logits = self.forward(X)
        logp = _log_softmax(logits)
        n = X.shape[0]
        loss = float(-logp[np.arange(n), labels].mean())
        dlogits = np.exp(logp)
        dlogits[np.arange(n), labels] -= 1.0
        dlogits /= n
        return loss, self._backprop_logits(X, h, dlogits)


class Adam:
    """Adam, operating in place on a parameter dict."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, param in self.params.items():
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
