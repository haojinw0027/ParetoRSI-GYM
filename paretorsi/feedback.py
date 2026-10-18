"""workspace/feedback/: everything the run has measured, as files the agent can
open.  Rebuilt from the run's records after every round; gold never appears.

    README.md                 this list
    scores.tsv                proxy headline score per round and axis, M_0 first
    budget.json               the fixed limits of the run
    candidates.json           the candidate store by domain, channel and source
    round_NN/eval.json        per axis: metric, score, all metrics, change vs M_0
                              and vs the parent checkpoint, truncation, output length
    round_NN/per_item/<axis>.jsonl
                              one proxy item per line: id, solve_rate (share of the
                              k samples correct), the model's first answer, and the
                              question without its gold
    round_NN/train.json       recipe, parent, loss curve
    round_NN/commit.json      pool size, entries new vs kept relative to the parent
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import models
from .common import load_benchmark, read_json, read_jsonl, round_dir, strip_gold, write_json, write_jsonl
from .skill import parents

ANSWER_CHARS = 1500
FIELD_CHARS = 1200


def _clip(v, n):
    return v[:n] + " ...[clipped]" if isinstance(v, str) and len(v) > n else v


def build(run: Path, cfg: dict) -> None:
    run = Path(run)
    fb = run / "workspace" / "feedback"
    if fb.exists():
        shutil.rmtree(fb)
    fb.mkdir(parents=True)
    (fb / "README.md").write_text(__doc__.split("\n\n", 1)[1])
    doms = list(cfg["domains"])
    scores = {s["round"]: s for s in (read_json(f) for f in sorted((run / "scores").glob("round_*.json")))}
    parent_of = parents(run)

    rows = ["round\t" + "\t".join(doms) + "\tmean\tparent"]
    for t, s in sorted(scores.items()):
        par = "" if t == 0 else ("M_0" if parent_of[t] == 0 else f"r{parent_of[t]}")
        rows.append(f"{'M_0' if t == 0 else t}\t" + "\t".join(f"{v:.4f}" for v in s["vector"])
                    + f"\t{s['mean']:.4f}\t{par}")
    (fb / "scores.tsv").write_text("\n".join(rows) + "\n")

    b = cfg["budget"]
    write_json(fb / "budget.json", {
        "rounds": b["rounds"], "next_round": (read_json(run / "state.json") or {"round": 0})["round"] + 1,
        "max_scheduled_tokens_per_round": b["max_scheduled_tokens"], "max_entries_per_round": b["max_entries"],
        "round_wall_clock_s": b["round_wall_clock_s"], "plan_deadline_s": b["plan_deadline_s"],
        "closeout_s": b["closeout_s"], "hf_bytes": b["hf_bytes"], "tunable": cfg["train"]["tunable"],
        "parent_choice": models.mode(cfg)})

    store = read_jsonl(run / "store" / "candidates.jsonl")
    count = lambda key: {v: sum(1 for e in store if key(e) == v) for v in sorted({key(e) for e in store}, key=str)}
    write_json(fb / "candidates.json", {
        "n": len(store), "by_domain": count(lambda e: e["domain"]), "by_channel": count(lambda e: e["channel"]),
        "by_source": count(lambda e: e["provenance"].get("repo") or "derived"),
        "pulls": read_jsonl(run / "store" / "hf_log.jsonl")[-50:]})

    for t, s in scores.items():
        _round(fb / f"round_{t:02d}", run, cfg, t, s, scores, parent_of)


def _round(out: Path, run: Path, cfg: dict, t: int, s: dict, scores: dict, parent_of: dict) -> None:
    out.mkdir()
    rd = round_dir(run, t)
    par = scores.get(parent_of.get(t, 0), scores[0])
    axes = {}
    for i, dom in enumerate(cfg["domains"]):
        rep = read_json(rd / "eval" / dom / "report.json")
        axes[dom] = {"metric": f"{rep['metric']}@{rep['k']}", "score": rep["score"], "scores": rep["scores"],
                     "n": rep["n"], "change_vs_M0_pp": round(100 * (s["vector"][i] - scores[0]["vector"][i]), 1),
                     "change_vs_parent_pp": None if t == 0 else round(100 * (s["vector"][i] - par["vector"][i]), 1),
                     "truncation_rate": rep["truncation_rate"], "mean_output_tokens": rep["mean_output_tokens"],
                     "extraction": rep["extraction"]}
    write_json(out / "eval.json", {"round": t, "axes": axes, "note": (
        "score is the axis's headline metric; `scores` has all of them. A one- or two-point move "
        "is within decoding noise at this k; read per_item before believing it.")})

    (out / "per_item").mkdir()
    for dom in cfg["domains"]:
        rates = read_json(rd / "eval" / dom / "per_item.json")
        first = {r["id"]: r["samples"][0]["text"] for r in read_jsonl(rd / "eval" / dom / "samples.jsonl")}
        write_jsonl(out / "per_item" / f"{dom}.jsonl", (
            {"id": it["id"], "solve_rate": rates[it["id"]], "model_answer": _clip(first[it["id"]], ANSWER_CHARS),
             **{k: _clip(v, FIELD_CHARS) for k, v in strip_gold(it).items() if k != "id"}}
            for it in load_benchmark(cfg["domains"][dom]["proxy"])))
    if t == 0:
        return
    rep = read_json(rd / "train" / "report.json") or {}
    loss = [r["loss"] for r in read_jsonl(rd / "train" / "train_log.jsonl")]
    write_json(out / "train.json", {"recipe": s["train_config"], "parent": s["init_from"],
                                    "examples": rep.get("n_examples"), "train_tokens": rep.get("train_tokens"),
                                    "loss_first": loss[0] if loss else None, "loss_last": loss[-1] if loss else None,
                                    "loss_curve": loss[::max(1, len(loss) // 20)]})
    commit = read_json(rd / "commit.json")
    write_json(out / "commit.json", {k: commit[k] for k in
                                     ("n_entries", "n_new", "n_kept", "n_dropped", "by_domain", "scheduled_tokens", "init_from")})
