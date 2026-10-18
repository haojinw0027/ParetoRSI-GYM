"""What happens after the agent: audit, train, score, record.

Run state, all under <run>/:
    state.json              {"round": last finished round, "history": [one record per round]}
    scores/round_NN.json    proxy scores of M_NN (round 0 is M_0)
    results.tsv             one line per finished round
    round_NN/               selection.jsonl, train_config.json, plan.json, SUBMITTED   (from the gateway)
                            audit.json, commit.json, pool.jsonl                         (from here)
                            train/{state,model,report.json,train_log.jsonl}, eval/<domain>/...

Training and scoring run as subprocesses: one torchrun over the GPUs for
training, then one vLLM engine on the first GPU for scoring.  Both resume: a
round whose train/report.json exists is not retrained, and a domain whose
eval report exists is not resampled.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import models, submission
from .common import ROOT, read_json, read_jsonl, round_dir, write_json, write_jsonl
from .store import CandidateStore


def _run(argv: list[str], log: Path, gpus: str) -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]),
           "CUDA_VISIBLE_DEVICES": gpus}
    with log.open("a") as f:
        f.write(" ".join(argv) + "\n")
        f.flush()
        rc = subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    if rc != 0:
        raise RuntimeError(f"{argv[2] if len(argv) > 2 else argv[0]} failed (exit {rc}); see {log}")


def evaluate(run: Path, suite: str, model: str, out: Path, gpus: list[str]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "-m", "paretorsi.evaluate", "--config", str(Path(run) / "config.lock.yaml"),
          "--suite", suite, "--model", model, "--out", str(out)], out / "eval.log", gpus[0])
    return read_json(out / "eval.json")


def _all_scores(out: Path, cfg: dict) -> dict:
    return {d: read_json(out / d / "report.json")["scores"] for d in cfg["domains"]}


def _state(run: Path) -> dict:
    return read_json(Path(run) / "state.json") or {"round": 0, "history": []}


def _finish(run: Path, record: dict) -> None:
    st = _state(run)
    st["round"] = record["round"]
    st["history"].append(record)
    write_json(Path(run) / "state.json", st)


def measure_m0(run: Path, cfg: dict, gpus: list[str]) -> dict:
    """s_0: the base model on the proxies, through the same code path as every round."""
    ev = evaluate(run, "proxy", cfg["base_model"], round_dir(run, 0) / "eval", gpus)
    s0 = {"round": 0, "order": ev["order"], "vector": ev["vector"], "mean": ev["mean"],
          "scores": _all_scores(round_dir(run, 0) / "eval", cfg)}
    (Path(run) / "scores").mkdir(exist_ok=True)
    write_json(Path(run) / "scores" / "round_00.json", s0)
    return s0


def skip(run: Path, t: int, status: str, **detail) -> None:
    """Record a round that trained nothing (no submission, or a failed audit)."""
    _finish(run, {"round": t, "status": status, **detail})


def pareto(vec: list[float], earlier: list[list[float]]) -> dict:
    dominates = lambda a, b: all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))
    return {"non_dominated": not any(dominates(e, vec) for e in earlier),
            "dominates": [i for i, e in enumerate(earlier) if dominates(vec, e)]}


def execute(run: Path, cfg: dict, t: int, gpus: list[str]) -> dict | None:
    """Audit, train and score a submitted round.  Returns its scores, or None
    if the audit failed (the round is then recorded as rejected)."""
    run, rd = Path(run), round_dir(run, t)
    store = CandidateStore(run, cfg)
    res = submission.audit(run, cfg, store, read_jsonl(rd / "selection.jsonl"),
                           read_json(rd / "train_config.json"))
    write_json(rd / "audit.json", {k: v for k, v in res.items() if k not in ("ids", "reasons")})
    if not res["ok"]:
        skip(run, t, "rejected", problems=res["problems"])
        return None
    tc, parent = res["train_config"], res["init_from"]
    prev = submission.previous_ids(run, cfg, parent)
    write_json(rd / "commit.json", {
        "round": t, "init_from": parent, "n_entries": res["n_entries"], "n_new": res["n_new"],
        "n_kept": res["n_kept"], "n_dropped": len(prev - set(res["ids"])), "by_domain": res["by_domain"],
        "scheduled_tokens": res["scheduled_tokens"], "train_config": tc,
        "new": [{"id": c, "reason": res["reasons"][c]} for c in res["ids"] if c not in prev]})
    write_jsonl(rd / "pool.jsonl", store.materialize(res["ids"]))

    out = rd / "train"
    if not (out / "report.json").exists():
        init = models.state_dir(run, parent)
        _run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpus)}",
              "-m", "paretorsi.train", "--base", cfg["base_model"], "--data", str(rd / "pool.jsonl"),
              "--out", str(out), *(["--init-from", str(init)] if init else []),
              "--lr", str(tc["lr"]), "--lora-rank", str(tc["lora_rank"]), "--batch-size", str(tc["batch_size"]),
              "--epochs", str(tc["epochs"]), "--max-len", str(tc["max_len"]),
              "--warmup-steps", str(tc["warmup_steps"]), "--seed", str(cfg["train"]["seed"])],
             rd / "train.log", ",".join(gpus))
    ev = evaluate(run, "proxy", str(out / "model"), rd / "eval", gpus)

    earlier = [read_json(f)["vector"] for f in sorted((run / "scores").glob("round_*.json"))]
    scores = {"round": t, "order": ev["order"], "vector": ev["vector"], "mean": ev["mean"],
              "scores": _all_scores(rd / "eval", cfg), "init_from": parent, "train_config": tc,
              "pareto": pareto(ev["vector"], earlier)}
    write_json(run / "scores" / f"round_{t:02d}.json", scores)
    _finish(run, {"round": t, "status": "ok", "init_from": parent, "train_config": tc,
                  "order": ev["order"], "vector": ev["vector"], "mean": ev["mean"],
                  "n_entries": res["n_entries"], "scheduled_tokens": res["scheduled_tokens"]})
    tsv = run / "results.tsv"
    if not tsv.exists():
        tsv.write_text("round\tparent\tn_entries\tscheduled_tokens\t" + "\t".join(ev["order"])
                       + "\tmean\tnon_dominated\n")
    with tsv.open("a") as f:
        f.write(f"{t}\t{parent}\t{res['n_entries']}\t{res['scheduled_tokens']}\t"
                + "\t".join(f"{v:.4f}" for v in ev["vector"])
                + f"\t{ev['mean']:.4f}\t{scores['pareto']['non_dominated']}\n")
    return scores
