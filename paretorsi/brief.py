"""The text the agent reads: BRIEF.md, CHANNELS.md, and the launch prompts.
Every number in it comes from the run's config."""

from __future__ import annotations

from pathlib import Path

from . import models
from .common import benchmark_path


def brief(cfg: dict, round_no: int) -> str:
    b, k = cfg["budget"], cfg["decoding"]["k"]
    doms = list(cfg["domains"])
    proxies = "\n".join(f"- **{d}**: {benchmark_path(cfg['domains'][d]['proxy']).name}, "
                        f"scored by {cfg['domains'][d]['proxy']['metric']}@{k}" for d in doms)
    tun = cfg["train"]["tunable"]
    free = [n for n, s in tun.items() if "choices" in s or s["min"] != s["max"]]
    pinned = [f"`{n}`={s['default']}" for n, s in tun.items() if n not in free]
    if models.mode(cfg) == "agent":
        parent = ("You choose the checkpoint each round trains from: `\"init_from\"` in "
                  "`train_config.json` names `base` or any `round_NN` in the model pool (`pb status`); "
                  "omitted, it is the newest. A non-base parent fixes `lora_rank` to its own. Training is "
                  "cumulative along the lineage you pick; branching from an earlier checkpoint discards "
                  "whatever later rounds taught, good and bad. An entry already on the parent's lineage "
                  "is a further epoch for it; anything else is new.")
    else:
        parent = ("Every round trains a fresh LoRA from the base model; only the candidate store and "
                  "your skill carry over.")
    t = f"round_{round_no:02d}"
    return f"""# ParetoRSI-Gym: round {round_no} of {b['rounds']}

You curate post-training data for `{cfg['base_model']}`. Each round you choose a
training pool and a recipe; the harness trains one LoRA pass over exactly that
pool and scores the result on {len(doms)} proxy evaluations:

{proxies}

Every item is answered {k} times. avg@{k} is the share of correct answers averaged
over items; maj@{k} counts an item when the gold letter is the unique most frequent
answer. A one- or two-point move is within decoding noise; check `solve_rate` in
the per-item files before believing it.

Your objective is the whole vector: move the Pareto front of the {len(doms)} scores
outward from the base model. A gain on one axis bought with a loss on another is
recorded as exactly that.

The proxies are your feedback, not the result. The run is scored afterwards on
held-out sets you never see and that share no item with the proxies, so what
counts is capability; fitting a proxy's surface form buys nothing there. You may
read the proxy questions and your model's answers (`feedback/round_NN/per_item/`,
`pb proxy peek --wrong`). Gold answers are never available. Training on an
evaluation set, its mirrors or solutions to its items is not allowed: their
source repositories are refused at pull, any entry sharing a {cfg['decontamination']['ngram']}-gram with
an evaluation item is rejected at registration, and web searches containing
proxy text are recorded as violations.

## A round

1. Read `feedback/README.md` and your skill, `.claude/skills/data-curation/SKILL.md`.
   In round 1 the skill is empty: fill every required slot, tagged `prior`.
2. Look at what you already have: `pb candidates`, `pb candidates stats --by meta.<field>`,
   `pb show <id>`, `pb proxy peek <axis>`. **{b['plan_deadline_s'] // 60} minutes into the round
   these refuse until `{t}/plan.json` exists.**
3. Write `{t}/plan.json`, a list of steps
   `{{"action": "...", "because": ["<principle id>", ...], "expect": {{"<axis>": "+2 to +4"}}}}`.
   `because` cites ids of active bullets in your skill. Acquisition refuses until the plan exists.
4. Acquire: `pb hf search`, `pb hf schema`, `pb hf peek`, `pb hf pull`. Pulled rows are
   registered on arrival and the reply lists their ids. **In the last {b['closeout_s'] // 60} minutes
   remote acquisition is closed**; selection and submission stay open.
5. Write `{t}/selection.jsonl` (`{{"id": ..., "reason": ...}}` per line; a reason is
   required for every entry new to the parent) and `{t}/train_config.json`.
6. Revise the skill: cite evidence as `rN:axis+points`, retire what was falsified, and run
   `pb skill check`. A skill that fails the check at round end is reverted.
7. `pb submit {t}/selection.jsonl --train-config {t}/train_config.json`, then exit.

## Limits

- Pool: at most {b['max_entries']:,} entries and {b['max_scheduled_tokens']:,} scheduled tokens
  (sum over entries of min(tokens, max_len), times epochs), counted with the base model's
  tokenizer. `pb submit` refuses a pool over either cap and tells you by how much.
- Recipe keys you may set: {', '.join(f'`{n}`' for n in free)}{' (fixed: ' + ', '.join(pinned) + ')' if pinned else ''}.
  The training seed is fixed.
- {parent}
- Time: {b['round_wall_clock_s'] // 60} minutes, hard. `pb status` shows what is left. Hugging Face
  bytes: {b['hf_bytes'] / 1e9:.0f} GB for the run.
"""


def channels(cfg: dict) -> str:
    return f"""# Channels and provenance

| channel | how a row gets in | provenance |
|---|---|---|
| `hf` | `pb hf pull --domain d --instruction-field q --output-field a` registers each mapped row | repo, config, split, row |
| `derived` | `pb register entries.jsonl`: your rewrite of a stored entry | parent_id, transform |

A `derived` line is `{{"channel": "derived", "domain": ..., "instruction": ..., "output": ...,
"provenance": {{"parent_id": "<stored id>", "transform": "<what you changed>"}}}}`.

`output` is what the model is trained to produce. Every entry passes
{cfg['decontamination']['ngram']}-gram decontamination against every proxy and held-out item;
rejected entries are listed with the reason in the reply.
"""


def launch_prompt(cfg: dict, round_no: int) -> str:
    b = cfg["budget"]
    return (f"This is round {round_no}. Read BRIEF.md, feedback/README.md, your skill at "
            f".claude/skills/data-curation/SKILL.md, and CHANNELS.md. Work in round_{round_no:02d}/. "
            f"You have {b['round_wall_clock_s'] // 60} minutes, after which the round ends with or without "
            f"a submission; `pb status` shows the time left. Finish by updating the skill and running `pb submit`.")


def relaunch_prompt() -> str:
    return ("You exited without running `pb submit`. Do not start new acquisition: make sure the skill's "
            "required slots are filled, write selection.jsonl from candidates already in the store "
            "(`pb candidates`), and submit now.")


def write_static(run: Path, cfg: dict) -> None:
    (Path(run) / "workspace" / "CHANNELS.md").write_text(channels(cfg))
