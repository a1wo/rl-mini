"""RLVR on a small LLM: Qwen3-0.6B on GSM8K, via modal-dojo.

Same four steps as experiments/mnist_grpo.py. What changes is who does the work:

    rollout      an SGLang server on Modal generates K completions per prompt
    reward       rl_mini/rewards.py:gsm8k_reward, shipped into the rollout workers
    advantage    advantage_estimator="grpo" inside slime
    update       Megatron on Modal GPUs

You write the reward function. The library owns everything else.

Prerequisites:
    uv sync --extra llm
    modal setup          # or: modal token new
    modal-dojo setup     # provisions the dashboard

Cost safety: the default preset is a 2-step smoke test and the script refuses to
launch without --yes, printing the rollout budget first. Nothing here starts a GPU
until you confirm.

Usage:
    uv run experiments/llm_gsm8k.py --dry-run          # print the plan, launch nothing
    uv run experiments/llm_gsm8k.py --yes              # smoke test (2 steps)
    uv run experiments/llm_gsm8k.py --preset short --yes
    uv run experiments/llm_gsm8k.py --watch --yes      # poll until the run finishes
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NOTE: imported from its own module, not defined here. modal-dojo ships the file a
# reward function lives in to the remote image as a flat module, so the reward must
# be self-contained -- see the docstring of rl_mini/rewards.py.
from rl_mini.rewards import gsm8k_reward

# Columns per https://huggingface.co/datasets/skrishna/gsm8k_only_answer
HF_REPO = "skrishna/gsm8k_only_answer"
INPUT_COLUMN = "text"
OUTPUT_COLUMN = "label"

PROMPT_TEMPLATE = (
    "{input}\n\n"
    "Solve the problem. Put only the final numeric answer inside \\boxed{{}}."
)

PRESETS = {
    # (num_rollout, rollout_batch_size, n_samples_per_prompt, global_batch_size, n_train)
    "smoke": (2, 4, 4, 16, 32),
    "short": (30, 16, 8, 128, 512),
    "real": (150, 32, 8, 256, 4000),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    p.add_argument("--gpu-type", default=None, help="override the recipe default")
    p.add_argument("--gpus", type=int, default=None, help="actor_num_gpus_per_node")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-response-len", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    p.add_argument("--yes", action="store_true", help="confirm GPU spend and launch")
    p.add_argument("--watch", action="store_true", help="poll until the run finishes")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace):
    """Construct the modal-dojo TrainConfig. Imports live here so --dry-run works
    (and so the smoke test fails fast with a readable message if the extra is
    not installed)."""
    from modal_dojo import (
        HuggingFaceDataset,
        Qwen3_0_6B,
        Qwen3_0_6B_Recipe,
        TrainConfig,
    )

    num_rollout, rollout_batch_size, n_samples, global_batch, n_train = PRESETS[args.preset]

    dataset = HuggingFaceDataset(
        HF_REPO,
        hf_split=f"train[:{n_train}]",
        input_column=INPUT_COLUMN,
        output_column=OUTPUT_COLUMN,
        input_format="text",
        prompt_template=PROMPT_TEMPLATE,
    )

    recipe_kwargs = dict(
        custom_rm_function=gsm8k_reward,
        advantage_estimator="grpo",
        num_rollout=num_rollout,
        rollout_batch_size=rollout_batch_size,
        n_samples_per_prompt=n_samples,
        global_batch_size=global_batch,
        rollout_max_response_len=args.max_response_len,
        rollout_temperature=args.temperature,
        save_interval=max(1, num_rollout // 2),
        # Qwen3 emits <think> blocks by default; disabling keeps smoke-test
        # responses short and the reward signal about math rather than length.
        apply_chat_template_kwargs='{"enable_thinking": false}',
    )
    if args.gpu_type:
        recipe_kwargs["gpu_type"] = args.gpu_type
    if args.gpus:
        recipe_kwargs["actor_num_gpus_per_node"] = args.gpus
    if args.lr is not None:
        recipe_kwargs["lr"] = args.lr

    return TrainConfig(
        model=Qwen3_0_6B(),
        dataset=dataset,
        recipe=Qwen3_0_6B_Recipe(**recipe_kwargs),
    )


def describe(args: argparse.Namespace) -> int:
    num_rollout, rollout_batch_size, n_samples, global_batch, n_train = PRESETS[args.preset]
    total = num_rollout * rollout_batch_size * n_samples
    print(f"preset               {args.preset}")
    print(f"model                Qwen3-0.6B")
    print(f"dataset              {HF_REPO} (train[:{n_train}])")
    print(f"reward               rl_mini.rewards.gsm8k_reward (0.1 format + 0.9 correct)")
    print(f"advantage estimator  grpo")
    print(f"num_rollout          {num_rollout}")
    print(f"rollout_batch_size   {rollout_batch_size}")
    print(f"n_samples_per_prompt {n_samples}")
    print(f"global_batch_size    {global_batch}")
    print(f"total generations    {num_rollout} x {rollout_batch_size} x {n_samples} = {total:,}")
    print(f"gpu                  {args.gpu_type or 'recipe default (H100)'}")
    return total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    describe(args)

    if args.dry_run:
        print("\n--dry-run: nothing launched.")
        return 0
    if not args.yes:
        print("\nThis starts GPUs on Modal and costs money. Re-run with --yes to launch.")
        return 1

    config = build_config(args)
    run = config.launch()
    print(f"\nrun id   {run.training_run_id}")
    print(f"modal app {run.modal_app_url}")
    print("dashboard: https://dojo.modal.dev  (or: modal-dojo run get <run id>)")

    if args.watch:
        checkpoint = None
        while True:
            done = run.done()
            latest = run.latest_checkpoint()
            if latest is not None and latest != checkpoint:
                checkpoint = latest
                print(f"new checkpoint: {checkpoint.path}")
            if done:
                break
            time.sleep(30)
        print("run finished" + (f"; last checkpoint {checkpoint.path}" if checkpoint else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
