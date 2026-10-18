#!/usr/bin/env python3
"""pb: the agent's interface to the gateway.  Prints JSON; exits non-zero on errors.

WHERE YOU STAND
  pb status                       time left, plan and closeout clocks, store, limits, model pool
  pb scores                       every scored round
  pb skill check                  validate your skill now rather than at round end

LOOK (refused after the plan deadline until round_NN/plan.json exists)
  pb proxy peek <axis> [--n 10] [--wrong]      proxy items without gold; --wrong: solve_rate < 0.5
  pb proxy stats <axis> [--by field]
  pb candidates [--domain d] [--channel c] [--n 50] [--where "meta.subject==physics"]
                [--in-pool N] [--preview-chars 160]  what is in the store; --in-pool N: round N's pool
  pb candidates stats [--domain d] [--by field] [--in-pool N]
  pb show <id>                                  one stored entry in full

ACQUIRE (pull and register need plan.json; Hugging Face routes close in the closeout window)
  pb hf search <query> [--limit 20]
  pb hf schema <repo> [--config c] [--split train] [--n 300]
  pb hf peek <repo> [--config c] [--split train] [--n 3]
  pb hf pull <repo> --domain d --instruction-field f --output-field f --n 1000
             [--config c] [--split train] [--offset 0] [--where clause]... [--shuffle-seed s]
                                                where ops: == != >= <= > < ~ (contains) !~
  pb register <entries.jsonl>                   derived entries you wrote (see CHANNELS.md)

FINISH
  pb submit <selection.jsonl> [--train-config train_config.json]
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def call(method, path, body=None, timeout=3600):
    url = os.environ["PB_GATEWAY_URL"].rstrip("/") + path
    req = urllib.request.Request(url, method=method, headers={"Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.stderr.write(e.read().decode(errors="replace") + "\n")
        sys.exit(1)


def query(**kw):
    pairs = [(k, v) for k, vs in kw.items() for v in (vs if isinstance(vs, list) else [vs]) if v not in (None, False)]
    return "?" + urllib.parse.urlencode(pairs) if pairs else ""


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().split("\n") if line.strip()]


def main():
    ap = argparse.ArgumentParser(prog="pb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("scores")
    sub.add_parser("skill").add_argument("what", choices=["check"])
    p = sub.add_parser("proxy")
    p.add_argument("what", choices=["peek", "stats"])
    p.add_argument("axis")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--wrong", action="store_true")
    p.add_argument("--by")
    p = sub.add_parser("candidates")
    p.add_argument("what", nargs="?", choices=["stats"])
    p.add_argument("--domain")
    p.add_argument("--channel")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--where", action="append", default=[])
    p.add_argument("--in-pool", type=int)
    p.add_argument("--preview-chars", type=int, default=160)
    p.add_argument("--by")
    sub.add_parser("show").add_argument("id")
    hf = sub.add_parser("hf").add_subparsers(dest="hf_cmd", required=True)
    s = hf.add_parser("search")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=20)
    for name, n in (("schema", 300), ("peek", 3), ("pull", None)):
        s = hf.add_parser(name)
        s.add_argument("repo")
        s.add_argument("--config")
        s.add_argument("--split", default="train")
        s.add_argument("--data-files")
        s.add_argument("--n", type=int, default=n, required=n is None)
        if name == "pull":
            s.add_argument("--domain", required=True)
            s.add_argument("--instruction-field", required=True)
            s.add_argument("--output-field", required=True)
            s.add_argument("--offset", type=int, default=0)
            s.add_argument("--where", action="append", default=[])
            s.add_argument("--shuffle-seed", type=int)
            s.add_argument("--scan-limit", type=int, default=200_000)
    sub.add_parser("register").add_argument("entries")
    p = sub.add_parser("submit")
    p.add_argument("selection")
    p.add_argument("--train-config")
    a = ap.parse_args()

    if a.cmd in ("status", "scores"):
        r = call("GET", "/" + a.cmd)
    elif a.cmd == "skill":
        r = call("GET", "/skill/check")
    elif a.cmd == "proxy":
        r = call("GET", f"/proxy/{a.what}" + query(axis=a.axis, n=a.n, wrong="1" if a.wrong else None, by=a.by))
    elif a.cmd == "candidates":
        q = (query(domain=a.domain, by=a.by, in_pool=a.in_pool) if a.what == "stats" else
             query(domain=a.domain, channel=a.channel, n=a.n, where=a.where, in_pool=a.in_pool,
                   preview_chars=a.preview_chars))
        r = call("GET", "/candidates" + ("/stats" if a.what else "") + q)
    elif a.cmd == "show":
        r = call("POST", "/show", {"id": a.id})
    elif a.cmd == "hf":
        body = {k: v for k, v in vars(a).items() if k not in ("cmd", "hf_cmd")}
        r = call("POST", f"/hf/{a.hf_cmd}", body)
    elif a.cmd == "register":
        r = call("POST", "/register", {"entries": read_jsonl(a.entries)})
    else:
        tc = json.load(open(a.train_config)) if a.train_config else None
        r = call("POST", "/submit", {"selection": read_jsonl(a.selection), "train_config": tc})
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if isinstance(r, dict) and r.get("ok") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
