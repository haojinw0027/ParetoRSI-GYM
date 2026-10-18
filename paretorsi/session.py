"""Run the curation loop.

    python -m paretorsi.session init --run runs/<id> --agent claude-code [--model M] [--seed 0] [--gpus 0,1]
    python -m paretorsi.session loop --run runs/<id>        # every remaining round
    python -m paretorsi.session round --run runs/<id>       # one round

`init` locks the config, prepares the agent's home, and measures M_0 on the
proxies.  A round:

  1. writes BRIEF.md, refreshes the harness region of the skill and records the
     skill as issued;
  2. starts the gateway and the agent; an agent that exits without submitting
     is relaunched into the same conversation while time is left;
  3. stops the gateway, audits the transcript, and checks the skill (a skill
     that fails is reverted to the issued version);
  4. archives the agent's scratch directory, then audits, trains and scores the
     submission (round.execute) and rebuilds feedback/.

A round already submitted when `round` starts (an earlier attempt died in
training or scoring) goes straight to step 4.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

from . import __version__, agents, brief, feedback, gateway, skill
from . import round as rnd
from .common import ROOT, load_config, read_json, round_dir, write_json
from .store import eval_ngrams


def init(a) -> None:
    run = Path(a.run).resolve()
    if run.exists() and any(run.iterdir()):
        raise SystemExit(f"{run} is not empty")
    cfg = load_config(Path(a.config))
    cfg["train"]["seed"] = a.seed
    (run / "workspace").mkdir(parents=True)
    (run / "config.lock.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cfg = load_config(run / "config.lock.yaml")
    write_json(run / "meta.json", {
        "version": __version__, "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "agent": a.agent, "model": a.model, "codex_effort": a.codex_effort, "gpus": a.gpus.split(","),
        "sandbox": not a.no_sandbox, "sandbox_ro": a.sandbox_ro, "cheetah_python": a.cheetah_python})
    agents.install_pb(run)
    agents.prepare(run, a.agent, codex_effort=a.codex_effort, cheetah_runner=a.cheetah_runner)
    print(f"measuring M_0 ({cfg['base_model']}) on the proxies", flush=True)
    s0 = rnd.measure_m0(run, cfg, a.gpus.split(","))
    write_json(run / "state.json", {"round": 0, "history": []})
    skill.path(run).parent.mkdir(parents=True)
    skill.path(run).write_text(skill.initial(cfg, skill.score_history(run)))
    (run / "principles").mkdir()
    shutil.copy2(skill.path(run), run / "principles" / "round_00.md")
    brief.write_static(run, cfg)
    feedback.build(run, cfg)
    print(f"initialised {run}: M_0 = {dict(zip(s0['order'], s0['vector']))}")


def run_round(run: Path) -> None:
    cfg = load_config(run / "config.lock.yaml")
    meta = read_json(run / "meta.json")
    gpus = meta["gpus"]
    t = read_json(run / "state.json")["round"] + 1
    if t > cfg["budget"]["rounds"]:
        raise SystemExit("all rounds are done")
    rd = round_dir(run, t)
    rd.mkdir(exist_ok=True)
    if not (rd / "SUBMITTED").exists():
        _curate(run, cfg, meta, t)
    if (rd / "SUBMITTED").exists():
        rnd.execute(run, cfg, t, gpus)
    else:
        rnd.skip(run, t, "no_submission")
    feedback.build(run, cfg)
    s = read_json(run / "scores" / f"round_{t:02d}.json")
    print(f"round {t}: " + (json.dumps(dict(zip(s["order"], s["vector"]))) if s else
                            read_json(run / "state.json")["history"][-1]["status"]), flush=True)


def _curate(run: Path, cfg: dict, meta: dict, t: int) -> None:
    b, rd = cfg["budget"], round_dir(run, t)
    ws = run / "workspace"
    (ws / f"round_{t:02d}").mkdir(exist_ok=True)
    (ws / "BRIEF.md").write_text(brief.brief(cfg, t))
    skill.refresh(run, cfg)
    issued = run / "principles" / f"issued_{t:02d}.md"
    shutil.copy2(skill.path(run), issued)

    start = time.time()
    (rd / "started_at").write_text(str(start))
    deadline = start + b["round_wall_clock_s"]
    srv, gw, url = gateway.serve(run, cfg, t, os.environ.get("HF_TOKEN"))
    launches = []
    try:
        for attempt in range(1 + b["max_relaunches"]):
            if (rd / "SUBMITTED").exists() or time.time() > deadline - 30:
                break
            rc = agents.launch(run, t, meta["agent"], model=meta["model"],
                               prompt=brief.launch_prompt(cfg, t) if attempt == 0 else brief.relaunch_prompt(),
                               resume=t > 1 or attempt > 0,
                               usd=b["agent_usd"] if attempt == 0 else b["agent_relaunch_usd"],
                               deadline=deadline, gateway_url=url, attempt=attempt,
                               confine=meta["sandbox"], cheetah_python=meta["cheetah_python"],
                               extra_ro=meta["sandbox_ro"])
            launches.append({"attempt": attempt, "exit": rc, "elapsed_s": round(time.time() - start)})
    finally:
        srv.shutdown()
        srv.server_close()

    audit = agents.audit_transcript(run, t, eval_ngrams(cfg), int(cfg["decontamination"]["ngram"]))
    write_json(rd / "agent.json", {"launches": launches, "row_views": gw.views, **audit})
    if audit["lookup_violations"]:
        print(f"round {t}: {len(audit['lookup_violations'])} search(es) or command(s) carried evaluation "
              f"text; see {rd / 'agent.json'}", flush=True)

    _, issued_regions, _ = skill.parse(issued.read_text())
    rep = skill.validate(skill.path(run).read_text(), cfg=cfg, round_no=t, issued_regions=issued_regions,
                         history=skill.score_history(run), parent_of=skill.parents(run))
    write_json(rd / "skill_check.json", {"ok": rep.ok, "problems": rep.problems, "stats": rep.stats})
    if not rep.ok:
        shutil.copy2(issued, skill.path(run))
    shutil.copy2(skill.path(run), run / "principles" / f"round_{t:02d}.md")
    scratch = ws / f"round_{t:02d}"
    if scratch.exists():
        shutil.move(str(scratch), rd / "scratch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--run", required=True)
    p.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    p.add_argument("--agent", choices=agents.AGENTS, required=True)
    p.add_argument("--model", default=None, help="the agent's model")
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids; training uses all, scoring the first")
    p.add_argument("--codex-effort", default="medium")
    p.add_argument("--cheetah-runner", default=None)
    p.add_argument("--cheetah-python", default=sys.executable)
    p.add_argument("--sandbox-ro", action="append", default=[], help="extra read-only path for the agent")
    p.add_argument("--no-sandbox", action="store_true", help="run the agent unconfined (recorded)")
    for name in ("round", "loop"):
        sub.add_parser(name).add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "init":
        return init(a)
    run = Path(a.run).resolve()
    if a.cmd == "round":
        return run_round(run)
    while read_json(run / "state.json")["round"] < load_config(run / "config.lock.yaml")["budget"]["rounds"]:
        run_round(run)


if __name__ == "__main__":
    main()
