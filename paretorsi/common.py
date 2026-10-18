"""Config loading and the small file helpers every module uses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("proxy", "heldout")
TASKS = ("math", "mcq", "code")
METRICS = ("avg", "maj", "pass")

# Benchmark fields that carry the answer.  They never leave the harness.
GOLD_KEYS = ("answer", "label", "test", "test_imports", "assertions")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("base_model", "decoding", "train", "budget", "principles",
                "decontamination", "domains"):
        if key not in cfg:
            raise ValueError(f"{path}: missing {key!r}")
    for dom, spec in cfg["domains"].items():
        for suite in SUITES:
            ev = spec.get(suite)
            if not ev:
                raise ValueError(f"{path}: domain {dom!r} has no {suite!r} benchmark")
            if ev["task"] not in TASKS:
                raise ValueError(f"{path}: {dom}.{suite}.task must be one of {TASKS}")
            if ev["metric"] not in METRICS or (ev["metric"] == "maj" and ev["task"] != "mcq"):
                raise ValueError(f"{path}: {dom}.{suite}.metric {ev['metric']!r} is invalid")
            if not benchmark_path(ev).exists():
                raise FileNotFoundError(benchmark_path(ev))
    if cfg["train"].get("init", "agent") not in ("agent", "base"):
        raise ValueError(f"{path}: train.init must be 'agent' or 'base'")
    return cfg


def domains(cfg: dict) -> list[str]:
    return list(cfg["domains"])


def benchmark_path(ev: dict) -> Path:
    p = Path(ev["data"])
    return p if p.is_absolute() else ROOT / p


def load_benchmark(ev: dict) -> list[dict]:
    """Items sorted by id, so every consumer sees them in the same order."""
    return sorted(read_jsonl(benchmark_path(ev)), key=lambda r: str(r["id"]))


def strip_gold(item: dict) -> dict:
    return {k: v for k, v in item.items() if k not in GOLD_KEYS}


def read_jsonl(path: Path) -> list[dict]:
    # split on "\n" only: str.splitlines() also breaks on U+2028 and friends,
    # which json.dumps(ensure_ascii=False) writes unescaped inside strings
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def read_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, obj) -> None:
    """Atomic: a reader sees the old file or the new one, never a torn write."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def round_dir(run: Path, t: int) -> Path:
    return Path(run) / f"round_{t:02d}"
