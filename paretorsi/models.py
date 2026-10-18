"""The model pool: M_0 ("base") and every round that trained and was scored.

With train.init == "agent" the agent names the parent of each round with
"init_from" in train_config.json ("base" or "round_NN"; omitted means the
newest checkpoint).  A LoRA state resumes only at the rank it was saved with,
so a non-base parent fixes lora_rank.  The pool is read from the run's own
records, never kept separately, so it cannot disagree with what happened.
"""

from __future__ import annotations

from pathlib import Path

from .common import read_json, read_jsonl, round_dir


def mode(cfg: dict) -> str:
    return cfg["train"].get("init", "agent")


def history(run: Path) -> list[dict]:
    return (read_json(Path(run) / "state.json") or {"history": []})["history"]


def entries(run: Path) -> list[dict]:
    """base first, then each scored round, oldest first."""
    s0 = read_json(Path(run) / "scores" / "round_00.json") or {}
    pool = [{"name": "base", "round": 0, "parent": None, "lora_rank": None,
             "scores": dict(zip(s0.get("order", []), s0.get("vector", []))), "mean": s0.get("mean")}]
    for h in history(run):
        if h["status"] == "ok":
            pool.append({"name": f"round_{h['round']:02d}", "round": h["round"], "parent": h["init_from"],
                         "lora_rank": h["train_config"]["lora_rank"], "train_config": h["train_config"],
                         "scores": dict(zip(h["order"], h["vector"])), "mean": h["mean"]})
    return pool


def get(run: Path, name: str) -> dict | None:
    return next((e for e in entries(run) if e["name"] == name), None)


def newest(run: Path) -> str:
    return entries(run)[-1]["name"]


def check_parent(run: Path, name, lora_rank) -> list[str]:
    e = get(run, name) if isinstance(name, str) else None
    if e is None:
        return [f"init_from {name!r} is not in the model pool {[x['name'] for x in entries(run)]}"]
    if e["lora_rank"] is not None and e["lora_rank"] != lora_rank:
        return [f"{name} was trained at lora_rank {e['lora_rank']}; resuming it requires the same rank "
                f"(got {lora_rank}), or choose init_from: base"]
    return []


def state_dir(run: Path, name: str) -> Path | None:
    """Adapter + optimizer state to resume from; None for the base model."""
    return None if name == "base" else round_dir(run, int(name.split("_")[1])) / "train" / "state"


def model_dir(run: Path, cfg: dict, name: str) -> str:
    """What evaluation loads: the base model, or a round's merged weights."""
    return cfg["base_model"] if name == "base" else str(round_dir(run, int(name.split("_")[1])) / "train" / "model")


def lineage(run: Path, name: str) -> list[str]:
    by = {e["name"]: e for e in entries(run)}
    out = []
    while name in by and name not in out:
        out.append(name)
        name = by[name]["parent"]
    return out


def trained_ids(run: Path, name: str) -> set[str]:
    """Every entry trained on along the lineage of `name`: one more pass over
    these is a later epoch for that checkpoint, anything else is new to it."""
    ids = set()
    for label in lineage(run, name):
        if label != "base":
            ids.update(r["id"] for r in read_jsonl(round_dir(run, int(label.split("_")[1])) / "selection.jsonl"))
    return ids
