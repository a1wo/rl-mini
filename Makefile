.PHONY: setup test data mnist compare llm-dry clean

setup:          ## create .venv and install numpy + pytest
	uv sync --extra dev

setup-llm:      ## additionally install modal-dojo (needed only for the LLM run)
	uv sync --extra dev --extra llm

test:
	uv run pytest -v

data:           ## download MNIST into ./data (~11 MB, once)
	uv run python -c "from rl_mini.data import load_mnist; load_mnist(n_test=1); print('ok')"

mnist:          ## RL on MNIST with GRPO
	uv run experiments/mnist_grpo.py

compare:        ## GRPO vs REINFORCE vs no-baseline vs supervised
	uv run experiments/mnist_grpo.py --compare --log-json runs/compare.json

llm-dry:        ## print the LLM training plan without launching anything
	uv run experiments/llm_gsm8k.py --dry-run

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ runs
