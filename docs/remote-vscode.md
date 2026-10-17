# Running this repo on a remote box from VS Code

**First, the thing worth knowing before you bother:** neither experiment needs a
GPU on the machine you are editing from.

* `experiments/mnist_grpo.py` is numpy on CPU. It finishes in seconds on a laptop.
* `experiments/llm_gsm8k.py` runs the training on **Modal's** GPUs. Your machine
  only builds the config and calls `.launch()`. A rented GPU box adds nothing.

A remote box is worth it if you want to run something *other* than these two
things on it, or you want the repo to live next to other GPU work. Here is the
setup either way.

## 1. Add an SSH host entry

Given a connection string like:

```
ssh -p 46408 root@211.21.129.110 -L 8080:localhost:8080
```

put this in `~/.ssh/config` on your laptop:

```sshconfig
Host gpu-box
    HostName 211.21.129.110
    User root
    Port 46408
    # Forward a port if you run a web UI on the box (jupyter, tensorboard, ...).
    LocalForward 8080 localhost:8080
    # Keep the connection alive through VS Code's idle periods.
    ServerAliveInterval 30
    ServerAliveCountMax 6
    # Rented boxes get new host keys constantly; trust on first use.
    StrictHostKeyChecking accept-new
    # Uncomment if the provider gave you a specific key:
    # IdentityFile ~/.ssh/id_ed25519
```

Verify it outside VS Code first — if `ssh gpu-box` does not work, Remote-SSH will
not either:

```bash
ssh gpu-box
```

**On the first connection you will be asked:**

```
The authenticity of host '[211.21.129.110]:46408' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Type **`yes`** and press Enter — the whole word, not `y`. That records the key in
`~/.ssh/known_hosts`. If your provider shows the host fingerprint in its console,
compare it first. The `StrictHostKeyChecking accept-new` line above makes future
hosts accept silently.

## 2. Connect VS Code

1. Install the **Remote - SSH** extension (`ms-vscode-remote.remote-ssh`). It is in
   this repo's recommended extensions, so VS Code will offer it.
2. `F1` -> **Remote-SSH: Connect to Host** -> `gpu-box`. It reads `~/.ssh/config`,
   so the host appears by name.
3. VS Code installs its server into `~/.vscode-server` on the box (first connect
   takes a minute).
4. **File -> Open Folder** -> the path where you cloned this repo *on the box*.

Extensions must be installed again on the remote side — VS Code shows them under
"Local - Installed" vs "SSH: gpu-box" and gives you an **Install in SSH** button.
The Python extension is the one you need.

## 3. Set the repo up on the box

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/a1wo/rl-mini.git
cd rl-mini
uv sync --extra dev
uv run pytest
```

If the repo is private, authenticate on the box first — `gh auth login`, or
forward your local SSH agent by adding `ForwardAgent yes` to the host entry and
cloning over SSH instead of HTTPS.

## 4. Point VS Code at the interpreter

`.vscode/settings.json` in this repo already sets:

```json
"python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
```

`${workspaceFolder}` resolves on whichever side the folder is open, so the same
setting works locally and remotely with no edits. If the interpreter does not pick
up, `F1` -> **Python: Select Interpreter** -> `./.venv/bin/python`.

Then:

* **Run and Debug** (`F5`) has four configs, including a `--dry-run` LLM config
  that launches nothing.
* **Terminal -> Run Task** has `setup`, `test`, and `mnist: compare estimators`.
* The Testing sidebar discovers the pytest suite automatically.

## 5. Port forwarding

The `LocalForward` line handles the case where you start a server on the box
yourself. VS Code also forwards ports automatically when it detects something
listening, and lists them in the **Ports** panel — you rarely need to configure it.

Note that the modal-dojo dashboard is **not** on localhost: it is hosted on Modal
and `modal-dojo setup` prints its URL. No tunnel needed, and it works the same
whether you launched from your laptop or from the box.
