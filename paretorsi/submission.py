"""The check a submission must pass before anything is trained.

The gateway runs it when the agent calls `pb submit`, so a bad submission is
refused while the agent can still fix it; the round driver runs it again on the
files on disk and trains nothing if it fails.

Compute is fixed per round, not the recipe: the agent may set any tunable key
within its range, but sum over entries of min(tokens, max_len) x epochs must
stay under budget.max_scheduled_tokens, counted with the base model's own
tokenizer on exactly the token ids training will see.
"""

from __future__ import annotations

from pathlib import Path

from . import models, render
from .common import read_jsonl, round_dir


def recipe(cfg: dict, raw: dict) -> tuple[dict, list[str]]:
    """The agent's train_config over the defaults, with every key range-checked."""
    tunable = cfg["train"]["tunable"]
    tc = {k: spec["default"] for k, spec in tunable.items()}
    problems = []
    for k, v in raw.items():
        spec = tunable.get(k)
        if spec is None:
            problems.append(f"train_config key {k!r} is not tunable; tunable keys are {sorted(tunable)}")
        elif "choices" in spec:
            if v not in spec["choices"]:
                problems.append(f"{k} must be one of {spec['choices']}, got {v!r}")
            else:
                tc[k] = v
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not spec["min"] <= v <= spec["max"]:
            problems.append(f"{k} must be a number in [{spec['min']}, {spec['max']}], got {v!r}")
        elif isinstance(spec["default"], int) and v != int(v):
            problems.append(f"{k} must be an integer, got {v}")
        else:
            tc[k] = type(spec["default"])(v)
    return tc, problems


def previous_ids(run: Path, cfg: dict, parent: str) -> set[str]:
    """Entries the model has already been trained on: the parent's lineage when
    the agent picks parents, otherwise the previous round's pool."""
    if models.mode(cfg) == "agent":
        return models.trained_ids(run, parent)
    ok = [h["round"] for h in models.history(run) if h["status"] == "ok"]
    return {r["id"] for r in read_jsonl(round_dir(run, ok[-1]) / "selection.jsonl")} if ok else set()


def audit(run: Path, cfg: dict, store, selection: list, train_config: dict | None) -> dict:
    problems = []
    raw = dict(train_config or {})
    parent = raw.pop("init_from", None)
    tc, bad = recipe(cfg, raw)
    problems += bad

    ids, reasons = [], {}
    for i, line in enumerate(selection):
        if not isinstance(line, dict) or set(line) - {"id", "reason"}:
            problems.append(f"selection line {i}: must be an object with only 'id' and 'reason'")
        elif not store.has(line.get("id")):
            problems.append(f"selection line {i}: unknown candidate id {line.get('id')!r}")
        else:
            ids.append(line["id"])
            reasons[line["id"]] = str(line.get("reason") or "").strip()
    if not ids:
        problems.append("selection is empty")
    if len(ids) != len(set(ids)):
        problems.append(f"{len(ids) - len(set(ids))} duplicate ids in the selection")
    if len(ids) > cfg["budget"]["max_entries"]:
        problems.append(f"{len(ids):,} entries exceed the per-round cap of {cfg['budget']['max_entries']:,}")

    if models.mode(cfg) == "agent":
        parent = parent if parent is not None else models.newest(run)
        problems += models.check_parent(run, parent, tc["lora_rank"])
    elif parent is not None:
        problems.append("init_from is not accepted: this run trains every round from the base model")
    else:
        parent = "base"

    prev = previous_ids(run, cfg, parent)
    silent = [c for c in ids if c not in prev and not reasons[c]]
    if silent:
        problems.append(f"{len(silent)} newly added entries have no reason (first: {silent[0]})")

    scheduled = None
    if ids and not bad:
        tok = render.load_tokenizer(cfg["base_model"])
        scheduled = tc["epochs"] * sum(
            min(render.example_length(tok, e["instruction"], e["output"]), tc["max_len"])
            for e in store.materialize(ids))
        if scheduled > cfg["budget"]["max_scheduled_tokens"]:
            problems.append(f"{scheduled:,} scheduled training tokens exceed the per-round cap of "
                            f"{cfg['budget']['max_scheduled_tokens']:,}; drop entries or lower max_len")
    by_domain = {}
    for c in ids:
        d = store.get(c)["domain"]
        by_domain[d] = by_domain.get(d, 0) + 1
    return {"ok": not problems, "problems": problems[:30], "train_config": tc, "init_from": parent,
            "ids": ids, "n_entries": len(ids), "n_new": sum(c not in prev for c in ids),
            "n_kept": sum(c in prev for c in ids), "by_domain": by_domain, "scheduled_tokens": scheduled,
            "cap": cfg["budget"]["max_scheduled_tokens"], "reasons": reasons}
