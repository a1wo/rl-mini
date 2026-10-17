# rl-mini

Two RL experiments at opposite ends of the scale, running the *same* four-step loop:

| | `experiments/mnist_grpo.py` | `experiments/llm_gsm8k.py` |
|---|---|---|
| policy | 101k-parameter MLP | Qwen3-0.6B |
| action | one digit, 0-9 | a chain-of-thought completion |
| reward | is the digit right? | is the boxed answer right? |
| rollout | numpy categorical sample | SGLang server on Modal |
| update | hand-written policy gradient | Megatron via [modal-dojo](https://github.com/modal-projects/modal-dojo) |
| hardware | your laptop, CPU | 1x H100 on Modal |
| time to first result | **~9 seconds** | ~20 minutes |

The point of the pairing: RL on an LLM is not a different algorithm from RL on a
tiny classifier. It is the same update rule with an expensive rollout step. If you
can read `rl_mini/grpo.py` (60 lines, mostly docstring) you understand what slime
and Megatron are doing with your GPUs.

```
1. sample K actions per prompt from the current policy
2. score each action with a reward function          <- you write this
3. turn rewards into advantages relative to the group <- this is GRPO
4. policy-gradient step weighted by those advantages
```

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/a1wo/rl-mini.git
cd rl-mini
uv sync --extra dev
uv run pytest                              # 38 tests, no GPU, no network
uv run experiments/mnist_grpo.py --compare # downloads MNIST (11 MB), ~9 s
```

No torch, no CUDA, no account needed for any of the above.

## Experiment 1 — RL on MNIST (CPU, seconds)

The policy never sees a label. It sees an image, emits one digit, and is told only
whether that digit was right. Learning from that scalar is the entire problem RLVR
solves on an LLM.

```bash
uv run experiments/mnist_grpo.py                        # GRPO
uv run experiments/mnist_grpo.py --estimator reinforce  # shared baseline
uv run experiments/mnist_grpo.py --estimator none       # no baseline at all
uv run experiments/mnist_grpo.py --mode supervised      # cross-entropy ceiling
uv run experiments/mnist_grpo.py --compare              # all four, one table
```

Knobs are named exactly as modal-dojo names them, because they mean the same thing:
`--num-rollout`, `--rollout-batch-size`, `--n-samples-per-prompt`, `--global-batch-size`.

### Measured results

400 steps, 256 images/step, 8 samples/image, seed 0, on a laptop CPU:

```
  setting                      test acc    samples     sec
  --------------------------------------------------------
  rl/grpo                       0.834    819,200     0.8
  rl/reinforce                  0.833    819,200     0.7
  rl/none                       0.828    819,200     0.7
  supervised                    0.930    102,400     0.6
```

Three things in that table are worth more than the headline number:

**Reward-only learning costs you roughly 10 accuracy points and 8x the samples.**
Same network, same optimizer, same data. The only difference is that the supervised
run is handed the correct label while the RL runs get one bit of feedback per
guess. That gap is the price of RL, and it is why you reach for RL only when you
cannot write down the target — which is exactly the situation for "write a good
haiku" or "make the test suite pass".

**GRPO and REINFORCE perform the same here, and that is expected.** GRPO's
advantage over a shared baseline comes from per-prompt difficulty variation; on
single-digit MNIST with a binary reward there is little to exploit. Do not read
this table as "GRPO is pointless" — read it as "GRPO's benefit is specific, and you
should be able to say what it is before you attribute a win to it". The one-line
change in `--estimator` lets you test that claim rather than assume it.

**The `degenerate` column is the real story.** It reaches 0.93 by the end: 93% of
prompt groups score *identically* across all 8 samples, contributing exactly zero
gradient. An 83%-accurate policy sampled 8 times mostly produces 8 correct answers.
The run is not compute-bound, it is signal-bound — which is why training plateaus
long before supervised does.

### The failure this repo is actually designed to show

`tests/test_learning.py::test_peaked_initialization_starves_exploration` is a
controlled experiment: identical data, network, optimizer, reward, and step count.
The only variable is how deterministic the policy is *at initialization*.

| output-layer init | test accuracy | classes never predicted |
|---|---|---|
| He-scaled (`final_layer_scale=1.0`) | 0.877 | class 0 |
| near-uniform (`final_layer_scale=0.01`, default) | **1.000** | none |

A policy gradient can only learn about actions it samples. A sharply-peaked initial
policy never samples some actions, so they are never rewarded, so they stay
unsampled. Accuracy plateaus at a suspiciously round 7/8 and nothing in the loss
curve looks wrong. The supervised control on the same network reaches 100% either
way — this failure is specific to learning from rewards.

This is the same fact as the `degenerate` column, and the same fact as "your base
model must already succeed sometimes for RLVR to bootstrap".

## Experiment 2 — RLVR on a small LLM (Modal, GPU)

Qwen3-0.6B on GSM8K with a verifiable reward, via modal-dojo. Training runs on
Modal's GPUs; your machine only launches it.

```bash
uv sync --extra dev --extra llm
modal setup            # one-time: authenticate Modal
modal-dojo setup       # one-time: provision the dashboard

uv run experiments/llm_gsm8k.py --dry-run     # print the plan, launch nothing
uv run experiments/llm_gsm8k.py --yes         # smoke test: 2 steps, 32 generations
uv run experiments/llm_gsm8k.py --preset short --yes
```

The script refuses to start GPUs without `--yes`, and prints the rollout budget
first:

```
preset               smoke
total generations    2 x 4 x 4 = 32
gpu                  recipe default (H100)
```

Presets are `smoke` (2 steps, proves the plumbing), `short` (30 steps, ~4k
generations), `real` (150 steps, ~38k generations). Cost scales as
`num_rollout x rollout_batch_size x n_samples_per_prompt`.

### The reward function is the whole job

```python
async def gsm8k_reward(args, sample, **kwargs) -> float:
    return gsm8k_score(sample.response or "", sample.label)
```

The `async` and the `(args, sample, **kwargs)` signature are required by slime, not
stylistic. The score is shaped rather than binary:

* **0.1 for format** — producing a `\boxed{}` at all. Learnable in a few steps, and
  it gets the reward curve off the floor while correctness is still near zero.
* **0.9 for correctness** — the boxed value matching the label, falling back to the
  last number in the response so a correct answer in the wrong format still scores.

Binary rewards are the default mistake. With 0/1 correctness on a model that is
rarely correct, nearly every group is degenerate (see the MNIST `degenerate`
column) and training learns nothing for a long time.

### One non-obvious constraint, which shapes this repo's layout

modal-dojo ships a `custom_rm_function` to the training cluster by copying **the
single file the function is defined in** to `/root/<module>.py` in the remote image
and importing it as a flat top-level module ([`ship_callable`](https://github.com/modal-projects/modal-dojo/blob/main/modal_dojo/common/launcher_helpers.py)).
The rest of your package does not come along.

So `rl_mini/rewards.py` imports **only the standard library** — no relative
imports, no sibling modules. That one rule is why the same code can be unit-tested
locally with no GPU and no Modal account (`tests/test_rewards.py`, 7 tests,
instant) and then run verbatim inside the rollout workers. Extra pip dependencies
go in via `image_overlay=lambda image: image.pip_install(...)`.

## Layout

```
rl_mini/
  grpo.py      group-relative advantages (this is GRPO, and it is 6 lines)
  policy.py    MLP policy + hand-written gradients + Adam
  env.py       ClassificationEnv, BernoulliBandit
  rewards.py   LLM rewards -- stdlib only, shipped to Modal as a flat module
  data.py      MNIST loader, no torchvision
experiments/
  mnist_grpo.py   CPU, seconds
  llm_gsm8k.py    Modal, GPU, costs money (gated behind --yes)
tests/            38 tests: gradient checks, advantage math, learning, rewards
```

## Tests

```bash
uv run pytest -v
```

Everything runs on CPU in a few seconds, with no network and no Modal account.

* `test_policy_grad.py` — every analytic gradient checked against finite
  differences, including the entropy bonus. The math is verified, not asserted.
* `test_grpo.py` — advantage math, including that all-correct and all-wrong groups
  both produce exactly zero signal.
* `test_learning.py` — the loop actually learns (bandit with a known optimum, and a
  synthetic classification task), plus the initialization-collapse experiment.
* `test_rewards.py` — the LLM reward function, offline.
* `test_mnist_experiment.py` — end-to-end smoke test, auto-skipped until MNIST is
  cached.

## Running on a remote GPU box via VS Code

See [docs/remote-vscode.md](docs/remote-vscode.md). Short version: you do **not**
need a GPU box for any of this — the MNIST experiment is CPU-only and the LLM
experiment runs on Modal's GPUs, not yours.

## License

MIT.
