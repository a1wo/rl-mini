"""RL on MNIST: the same GRPO loop as the LLM run, 100k parameters, CPU, ~15 seconds.

The policy never sees a label. It sees an image, emits one digit, and is told only
whether that digit was right. Learning from that scalar is the whole problem RLVR
solves on an LLM; here you can watch it happen in a terminal without a GPU.

The knobs are named exactly as modal-dojo names them, because they mean the same
things:

    num_rollout           number of training steps
    rollout_batch_size    prompts (images) sampled per step
    n_samples_per_prompt  completions (digit guesses) per prompt -> the GRPO group
    global_batch_size     samples per optimizer step

Usage:
    uv run experiments/mnist_grpo.py                       # GRPO
    uv run experiments/mnist_grpo.py --estimator reinforce # shared baseline
    uv run experiments/mnist_grpo.py --mode supervised     # cross-entropy ceiling
    uv run experiments/mnist_grpo.py --compare             # all three, one table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_mini.data import load_mnist
from rl_mini.env import ClassificationEnv
from rl_mini.grpo import group_advantages, group_reward_std, reinforce_advantages
from rl_mini.policy import Adam, MLPPolicy

HIDDEN = 128


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("rl", "supervised"), default="rl")
    p.add_argument(
        "--estimator",
        choices=("grpo", "grpo-unnormalized", "reinforce", "none"),
        default="grpo",
        help="'none' uses raw rewards as advantages (no baseline) - watch it struggle.",
    )
    p.add_argument("--num-rollout", type=int, default=400)
    p.add_argument("--rollout-batch-size", type=int, default=256)
    p.add_argument("--n-samples-per-prompt", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=None, help="limit training examples")
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--compare", action="store_true", help="run grpo/reinforce/supervised")
    p.add_argument("--log-json", type=Path, default=None, help="write per-step metrics")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def advantages_for(estimator: str, rewards: np.ndarray) -> np.ndarray:
    if estimator == "grpo":
        return group_advantages(rewards, normalize=True)
    if estimator == "grpo-unnormalized":
        return group_advantages(rewards, normalize=False)
    if estimator == "reinforce":
        return reinforce_advantages(rewards)
    if estimator == "none":
        return np.asarray(rewards, dtype=np.float64)
    raise ValueError(f"unknown estimator {estimator!r}")


def accuracy(policy: MLPPolicy, X: np.ndarray, y: np.ndarray) -> float:
    return float((policy.greedy_actions(X) == y).mean())


def train(data: dict[str, np.ndarray], args: argparse.Namespace, *, estimator: str, mode: str) -> dict:
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    rng = np.random.default_rng(args.seed)
    policy = MLPPolicy(X_train.shape[1], 10, hidden=HIDDEN, seed=args.seed)
    optimizer = Adam(policy.param_dict(), lr=args.lr)
    env = ClassificationEnv(n_actions=10)

    prompts_per_opt_step = max(1, args.global_batch_size // args.n_samples_per_prompt)
    history: list[dict] = []
    started = time.monotonic()
    label = f"{mode}" if mode == "supervised" else f"{mode}/{estimator}"

    for step in range(1, args.num_rollout + 1):
        index = rng.integers(0, X_train.shape[0], size=args.rollout_batch_size)
        X, y = X_train[index], y_train[index]

        if mode == "supervised":
            # Reference ceiling: identical network and optimizer, but handed the label.
            loss, grads = policy.ce_loss_and_grads(X, y)
            optimizer.step(grads)
            mean_reward = accuracy(policy, X, y)
            degenerate = 0.0
        else:
            # 1. rollout: K samples per prompt from the current policy
            actions = policy.sample_actions(X, args.n_samples_per_prompt, rng)
            # 2. environment scores each sample
            rewards = env.reward(actions, y)
            # 3. rewards -> advantages
            adv = advantages_for(estimator, rewards)
            # 4. policy-gradient step(s)
            loss = 0.0
            n_chunks = 0
            for start in range(0, args.rollout_batch_size, prompts_per_opt_step):
                stop = start + prompts_per_opt_step
                chunk_loss, grads = policy.pg_loss_and_grads(
                    X[start:stop],
                    actions[start:stop],
                    adv[start:stop],
                    entropy_coef=args.entropy_coef,
                )
                optimizer.step(grads)
                loss += chunk_loss
                n_chunks += 1
            loss /= max(n_chunks, 1)
            mean_reward = float(rewards.mean())
            degenerate = float((group_reward_std(rewards) == 0).mean())

        record = {"step": step, "loss": loss, "mean_reward": mean_reward, "degenerate_groups": degenerate}

        if step % args.eval_interval == 0 or step == args.num_rollout:
            record["test_accuracy"] = accuracy(policy, X_test, y_test)
            if not args.quiet:
                print(
                    f"[{label}] step {step:>4}/{args.num_rollout}  "
                    f"reward {mean_reward:.3f}  "
                    f"test_acc {record['test_accuracy']:.3f}  "
                    f"degenerate {degenerate:.2f}  "
                    f"loss {loss:+.4f}"
                )
        history.append(record)

    final_accuracy = accuracy(policy, X_test, y_test)
    return {
        "label": label,
        "mode": mode,
        "estimator": estimator if mode == "rl" else None,
        "final_test_accuracy": final_accuracy,
        "seconds": round(time.monotonic() - started, 2),
        "samples_consumed": args.num_rollout
        * args.rollout_batch_size
        * (args.n_samples_per_prompt if mode == "rl" else 1),
        "history": history,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_mnist(n_train=args.n_train, n_test=args.n_test)
    print(
        f"MNIST: {data['X_train'].shape[0]} train / {data['X_test'].shape[0]} test, "
        f"policy 784-{HIDDEN}-10"
    )

    if args.compare:
        runs = [
            train(data, args, estimator="grpo", mode="rl"),
            train(data, args, estimator="reinforce", mode="rl"),
            train(data, args, estimator="none", mode="rl"),
            train(data, args, estimator="grpo", mode="supervised"),
        ]
    else:
        runs = [train(data, args, estimator=args.estimator, mode=args.mode)]

    print("\n  setting                      test acc    samples     sec")
    print("  " + "-" * 56)
    for run in runs:
        print(
            f"  {run['label']:<26} {run['final_test_accuracy']:>8.3f} "
            f"{run['samples_consumed']:>10,} {run['seconds']:>7.1f}"
        )

    if args.log_json:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        args.log_json.write_text(json.dumps({"args": vars(args) | {"log_json": str(args.log_json)}, "runs": runs}, indent=2, default=str))
        print(f"\nwrote {args.log_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
