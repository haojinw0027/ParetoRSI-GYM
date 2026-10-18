# ParetoRSI-Gym

ParetoRSI-Gym evaluates whether an LLM agent can curate post-training data that improves a
model across several domains at once (math, science, medicine, finance, code). Each round the
agent searches Hugging Face for data, assembles a training pool within a fixed token budget,
picks a training recipe, and reads back per-domain proxy results; the harness trains a LoRA
on exactly that pool and scores it. After the last round, checkpoints are scored on held-out
benchmarks the agent never sees.

## Install

```bash
pip install -e ".[gpu]"        # vLLM, transformers, peft
export HF_TOKEN=...            # held by the harness, never passed to the agent
```

The benchmark files (`benchmarks/`) are not included in this release yet. The agent runs
inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox (`--no-sandbox` to
disable, which is recorded in the run's `meta.json`).

## Run

`init` measures the base model on the proxies, `loop` runs every round, `heldout` scores the
final and proxy-best checkpoints.

**Claude Code** (`claude` on `PATH`; `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` or a logged-in `~/.claude`)

```bash
python -m paretorsi.session init --run runs/claude --agent claude-code --model opus --gpus 0,1
python -m paretorsi.session loop --run runs/claude
```

**Codex** (`codex` on `PATH`; `OPENAI_API_KEY` or a logged-in `~/.codex`)

```bash
python -m paretorsi.session init --run runs/codex --agent codex --model gpt-5.6-sol --codex-effort medium --gpus 0,1
python -m paretorsi.session loop --run runs/codex
```

**CheetahClaw** (`DEEPSEEK_API_KEY`; a runner script that drives cheetahclaws, see `paretorsi/agents.py`)

```bash
python -m paretorsi.session init --run runs/cheetah --agent cheetahclaws --model deepseek/deepseek-v4-flash \
    --cheetah-runner path/to/cheetah_runner.py --gpus 0,1
python -m paretorsi.session loop --run runs/cheetah
```

**Held-out evaluation** (after the last round)

```bash
python -m paretorsi.heldout --run runs/claude
```

Settings that the agent cannot change (base model, budget, rounds, decoding, benchmarks) live in
`config/default.yaml`; `init` copies it into the run as `config.lock.yaml`.
