"""Smoke test for the MNIST script. Skipped automatically when MNIST isn't cached.

Run `make data` (or any MNIST experiment) once to populate ./data, and this test
starts exercising the real end-to-end path.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rl_mini.data import DEFAULT_DATA_DIR, FILES  # noqa: E402

pytestmark = pytest.mark.skipif(
    not all((DEFAULT_DATA_DIR / name).exists() for name in FILES.values()),
    reason="MNIST not downloaded; run `make data`",
)


def test_mnist_rl_runs_and_beats_chance():
    from experiments.mnist_grpo import main

    exit_code = main(
        [
            "--num-rollout", "60",
            "--rollout-batch-size", "128",
            "--n-samples-per-prompt", "8",
            "--n-train", "4000",
            "--n-test", "1000",
            "--eval-interval", "60",
            "--quiet",
        ]
    )
    assert exit_code == 0
