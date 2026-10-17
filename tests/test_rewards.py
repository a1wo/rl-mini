"""The LLM reward function, tested offline: no GPU, no Modal account, no network.

This is the main practical argument for keeping reward logic in a plain module:
the exact code that will run inside the rollout workers is unit-testable here.
"""

import asyncio
import types

import pytest

from rl_mini.rewards import (
    extract_boxed,
    extract_last_number,
    gsm8k_reward,
    gsm8k_score,
    numbers_match,
    strip_thinking,
)


def test_strip_thinking_handles_closed_and_truncated_blocks():
    assert strip_thinking("<think>hmm</think>The answer is 4") == "The answer is 4"
    assert strip_thinking("rambling</think>42") == "42"
    assert strip_thinking("no thinking here") == "no thinking here"


def test_extract_boxed_takes_the_last_box():
    assert extract_boxed(r"first \boxed{1} then \boxed{72}") == "72"
    assert extract_boxed("no box") is None


def test_extract_last_number():
    assert extract_last_number("he had 3 apples and bought 12 more") == "12"
    assert extract_last_number("total: 1,250.50") == "1,250.50"
    assert extract_last_number("nothing numeric") is None


@pytest.mark.parametrize(
    "predicted,label,expected",
    [
        ("72", "72", True),
        ("72.0", "72", True),
        ("1,250", "1250", True),
        ("$18", "18", True),
        ("71", "72", False),
        (None, "72", False),
        ("not a number", "72", False),
    ],
)
def test_numbers_match(predicted, label, expected):
    assert numbers_match(predicted, label) is expected


def test_score_is_shaped_not_binary():
    correct = gsm8k_score(r"Work it out. \boxed{72}", "72")
    boxed_but_wrong = gsm8k_score(r"\boxed{71}", "72")
    right_answer_no_box = gsm8k_score("so the answer is 72", "72")
    nothing = gsm8k_score("I don't know", "72")

    assert correct == pytest.approx(1.0)
    # Format credit alone keeps the curve off the floor early in training.
    assert boxed_but_wrong == pytest.approx(0.1)
    assert right_answer_no_box == pytest.approx(0.9)
    assert nothing == pytest.approx(0.0)
    assert nothing < boxed_but_wrong < right_answer_no_box < correct


def test_score_ignores_the_thinking_block():
    response = r"<think>maybe 71? no, 72</think>\boxed{72}"
    assert gsm8k_score(response, "72") == pytest.approx(1.0)


def test_reward_hook_matches_the_dojo_contract():
    """custom_rm_function must be async and take (args, sample, **kwargs)."""
    sample = types.SimpleNamespace(response=r"\boxed{72}", label="72", metadata={})
    score = asyncio.run(gsm8k_reward(None, sample))
    assert score == pytest.approx(1.0)
    # Metadata is what shows up per-rollout in the dashboard.
    assert sample.metadata["has_boxed"] is True
    assert sample.metadata["response_chars"] > 0


def test_reward_hook_survives_a_missing_response():
    sample = types.SimpleNamespace(response=None, label="72", metadata=None)
    assert asyncio.run(gsm8k_reward(None, sample)) == pytest.approx(0.0)
