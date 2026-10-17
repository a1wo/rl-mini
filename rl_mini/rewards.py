"""Verifiable reward functions for the LLM run.

IMPORTANT — why this file imports nothing from ``rl_mini``:

modal-dojo ships a ``custom_rm_function`` to the training cluster by copying *the
single file the function is defined in* to ``/root/<module>.py`` inside the remote
image and importing it as a flat top-level module (see ``ship_callable`` in
modal_dojo/common/launcher_helpers.py). The rest of your package does not come
along. So a reward module must be self-contained: standard library only, no
relative imports, no reaching into sibling modules.

That constraint is also why the scoring logic lives here as plain functions: the
same code is unit-tested locally in ``tests/test_rewards.py`` with no GPU, no Modal
account and no network, and then runs verbatim inside the rollout workers.
"""

from __future__ import annotations

import re

__all__ = [
    "strip_thinking",
    "extract_boxed",
    "extract_last_number",
    "numbers_match",
    "gsm8k_score",
    "gsm8k_reward",
]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"^.*?</think>", re.DOTALL)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

FORMAT_WEIGHT = 0.1
CORRECT_WEIGHT = 0.9


def strip_thinking(text: str) -> str:
    """Remove a Qwen-style ``<think>...</think>`` block, closed or truncated."""
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = _OPEN_THINK.sub("", text)
    return text.strip()


def extract_boxed(text: str) -> str | None:
    """Return the last ``\\boxed{...}`` payload, or None."""
    matches = _BOXED.findall(text)
    return matches[-1].strip() if matches else None


def extract_last_number(text: str) -> str | None:
    """Return the last number appearing in the text, or None."""
    matches = _NUMBER.findall(text)
    return matches[-1] if matches else None


def numbers_match(predicted: str | None, label: str | None, tol: float = 1e-6) -> bool:
    """Compare two numeric strings tolerantly (commas, trailing zeros, '$')."""
    if predicted is None or label is None:
        return False
    def clean(value: str) -> str:
        return value.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return abs(float(clean(predicted)) - float(clean(str(label)))) <= tol
    except (ValueError, TypeError):
        return False


def gsm8k_score(response: str, label: str | None) -> float:
    """Shaped reward for a grade-school math answer, in [0, 1].

    Two components, because a single 0/1 correctness bit gives GRPO nothing to
    work with until the model is already nearly right:

    * 0.1 for *format* — putting a final answer in ``\\boxed{}`` at all. This is
      learnable within a handful of steps and gets the reward curve off the floor.
    * 0.9 for *correctness* — the boxed value matching the label. Falls back to
      the last number in the response, so a model that reasons correctly but
      forgets the box still gets credit for the answer.
    """
    text = strip_thinking(response or "")
    boxed = extract_boxed(text)
    score = FORMAT_WEIGHT if boxed is not None else 0.0
    predicted = boxed if boxed is not None else extract_last_number(text)
    if numbers_match(predicted, label):
        score += CORRECT_WEIGHT
    return score


async def gsm8k_reward(args, sample, **kwargs) -> float:
    """modal-dojo reward hook.

    The ``async def`` and the ``(args, sample, **kwargs)`` signature are required by
    the underlying framework (slime), not stylistic. ``sample.response`` is the
    generated text; ``sample.label`` is the dataset's ``output_column``. Anything
    written to ``sample.metadata`` shows up per-rollout in the dashboard.
    """
    score = gsm8k_score(sample.response or "", sample.label)
    try:
        sample.metadata = {
            **(getattr(sample, "metadata", None) or {}),
            "has_boxed": extract_boxed(strip_thinking(sample.response or "")) is not None,
            "response_chars": len(sample.response or ""),
        }
    except Exception:  # noqa: BLE001 - never fail a rollout over telemetry
        pass
    return score
