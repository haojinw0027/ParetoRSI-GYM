"""Score a finished run on the held-out suite.

    python -m paretorsi.heldout --run runs/<id> [--gpus 0]

Checkpoints are chosen from proxy scores only, then scored on held-out:
    M_0          the base model
    Final        the last scored round
    Proxy Best   the round with the highest mean proxy score (earlier round on a tie)
    Domain Best  per domain, the round with the highest proxy score on that domain
                 (ties: higher proxy avg@k, then the earlier round); each domain
                 reports its own round's held-out score
M_0 and every chosen round are scored in this one invocation, so they share
software and hardware.  Writes <run>/heldout/<checkpoint>/ and summary.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import models
from . import round as rnd
from .common import load_config, read_json, write_json


def select(run: Path, cfg: dict) -> dict:
    k = cfg["decoding"]["k"]
    rounds = [read_json(f) for f in sorted((run / "scores").glob("round_*.json"))]
    trained = [s for s in rounds if s["round"] > 0]
    if not trained:
        raise SystemExit("no scored round to select from")
    pick = {"final": trained[-1]["round"],
            "proxy_best": max(trained, key=lambda s: (s["mean"], -s["round"]))["round"]}
    for i, d in enumerate(cfg["domains"]):
        pick[f"domain_best:{d}"] = max(trained, key=lambda s: (s["vector"][i], s["scores"][d][f"avg@{k}"],
                                                              -s["round"]))["round"]
    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpus", default=None, help="default: the run's own")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    cfg = load_config(run / "config.lock.yaml")
    gpus = a.gpus.split(",") if a.gpus else read_json(run / "meta.json")["gpus"]
    pick = select(run, cfg)
    results = {}
    for t in sorted({0, *pick.values()}):
        name = "base" if t == 0 else f"round_{t:02d}"
        results[t] = rnd.evaluate(run, "heldout", models.model_dir(run, cfg, name), run / "heldout" / name, gpus)

    doms = list(cfg["domains"])
    table = {"M_0": dict(zip(doms, results[0]["vector"])),
             "final": dict(zip(doms, results[pick["final"]]["vector"])),
             "proxy_best": dict(zip(doms, results[pick["proxy_best"]]["vector"])),
             "domain_best": {d: results[pick[f"domain_best:{d}"]]["vector"][i] for i, d in enumerate(doms)}}
    for row in table.values():
        row["mean"] = round(sum(row[d] for d in doms) / len(doms), 4)
    write_json(run / "heldout" / "summary.json", {"selected_rounds": pick, "scores": table,
                                                  "metric": {d: cfg["domains"][d]["heldout"]["metric"] for d in doms}})
    for name, row in table.items():
        print(f"{name:12s} " + "  ".join(f"{d} {100 * row[d]:5.2f}" for d in doms) + f"  mean {100 * row['mean']:5.2f}")


if __name__ == "__main__":
    main()
