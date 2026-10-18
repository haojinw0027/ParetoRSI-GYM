"""The gateway: a 127.0.0.1 HTTP service, the agent's only way to data.

It runs inside the harness process and holds the Hugging Face token; the agent
runs without it and reaches the gateway through `pb`.  Three clocks, all
measured from the round's start:

    plan_deadline_s   after it, read routes refuse until round_NN/plan.json exists
    (always)          pull and register refuse until plan.json exists and is valid
    closeout_s        in the final window, every Hugging Face route refuses

GET   /status /scores /skill/check
      /proxy/peek?axis=&n=&wrong=1     /proxy/stats?axis=&by=
      /candidates?domain=&channel=&n=&where=&in_pool=&preview_chars=
      /candidates/stats?domain=&by=&in_pool=
POST  /show {id}   /hf/search {query, limit}   /hf/schema /hf/peek {repo, config, split, n}
      /hf/pull {repo, config, split, n, offset, domain, instruction_field, output_field,
                where, shuffle_seed, data_files, scan_limit}
      /register {entries}   /submit {selection, train_config}
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import hub, models, skill, stats, submission
from .common import load_benchmark, read_json, read_jsonl, round_dir, strip_gold, write_json, write_jsonl
from .store import CandidateStore


class Refused(Exception):
    """A request the rules of the round do not allow (HTTP 403)."""


class Gateway:
    def __init__(self, run: Path, cfg: dict, round_no: int, hf_token: str | None):
        self.run, self.cfg, self.t = Path(run), cfg, round_no
        self.rd = round_dir(run, round_no)
        self.scratch = self.run / "workspace" / f"round_{round_no:02d}"
        self.started = float((self.rd / "started_at").read_text())
        self.store = CandidateStore(run, cfg)
        self.hub = hub.Hub(run, byte_cap=int(cfg["budget"]["hf_bytes"]), token=hf_token)
        self.views = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------ clocks
    def elapsed(self) -> float:
        return time.time() - self.started

    def gate_read(self) -> None:
        if not (self.scratch / "plan.json").exists() and self.elapsed() > self.cfg["budget"]["plan_deadline_s"]:
            raise Refused(f"the inspection window closed at {self.cfg['budget']['plan_deadline_s']}s; "
                          f"write round_{self.t:02d}/plan.json and reads resume")

    def gate_remote(self) -> None:
        b = self.cfg["budget"]
        if self.elapsed() > b["round_wall_clock_s"] - b["closeout_s"]:
            raise Refused("closeout: remote acquisition is closed for the rest of the round; "
                          "select from the store and submit")

    def plan(self) -> list:
        p = self.scratch / "plan.json"
        if not p.exists():
            raise Refused(f"write round_{self.t:02d}/plan.json before acquiring data")
        try:
            steps = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"plan.json is not valid JSON: {e}")
        if not isinstance(steps, list) or not steps:
            raise ValueError("plan.json must be a non-empty list of steps")
        _, _, bullets = skill.parse(skill.path(self.run).read_text())
        active = {b.id for b in bullets if b.slot != "retired"}
        problems = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict) or not str(s.get("action", "")).strip():
                problems.append(f"step {i} needs an 'action'")
                continue
            cited = s.get("because")
            if not isinstance(cited, list) or not cited or set(cited) - active:
                problems.append(f"step {i}: 'because' must list ids of active skill bullets "
                                f"(not active: {sorted(set(cited or []) - active)})")
            if set(s.get("expect") or {}) - set(self.cfg["domains"]):
                problems.append(f"step {i}: 'expect' names an unknown axis")
        if problems:
            raise ValueError("plan.json: " + "; ".join(problems))
        return steps

    # ------------------------------------------------------------ reads
    def status(self) -> dict:
        b = self.cfg["budget"]
        return {"round": self.t, "submitted": (self.rd / "SUBMITTED").exists(),
                "seconds_left": int(b["round_wall_clock_s"] - self.elapsed()),
                "seconds_left_to_plan": max(0, int(b["plan_deadline_s"] - self.elapsed())),
                "plan_written": (self.scratch / "plan.json").exists(),
                "seconds_until_closeout": max(0, int(b["round_wall_clock_s"] - b["closeout_s"] - self.elapsed())),
                "store": self.store.summary(), "row_views": self.views,
                "hf_bytes": {"used": self.hub.bytes_used, "cap": self.hub.byte_cap},
                "limits": {"max_entries": b["max_entries"], "max_scheduled_tokens": b["max_scheduled_tokens"]},
                "tunable": self.cfg["train"]["tunable"],
                "model_pool": models.entries(self.run) if models.mode(self.cfg) == "agent" else None}

    def scores(self) -> dict:
        return {"rounds": [read_json(f) for f in sorted((self.run / "scores").glob("round_*.json"))]}

    def skill_check(self) -> dict:
        _, issued, _ = skill.parse((self.run / "principles" / f"issued_{self.t:02d}.md").read_text())
        rep = skill.validate(skill.path(self.run).read_text(), cfg=self.cfg, round_no=self.t,
                             issued_regions=issued, history=skill.score_history(self.run),
                             parent_of=skill.parents(self.run))
        return {"ok": rep.ok, "problems": rep.problems, "stats": rep.stats}

    def _proxy_rows(self, axis: str) -> list[dict]:
        if axis not in self.cfg["domains"]:
            raise ValueError(f"axis must be one of {list(self.cfg['domains'])}")
        last = max(self.run.glob("round_*/eval/" + axis + "/per_item.json"), default=None,
                   key=lambda p: p.parent.parent.parent.name)
        rates = read_json(last) if last else {}
        return [{**strip_gold(it), "solve_rate": rates.get(it["id"])}
                for it in load_benchmark(self.cfg["domains"][axis]["proxy"])]

    def proxy_peek(self, axis: str, n: int, wrong: bool) -> dict:
        rows = [r for r in self._proxy_rows(axis) if not wrong or (r["solve_rate"] is not None and r["solve_rate"] < 0.5)]
        return {"axis": axis, "n_matching": len(rows), "items": rows[:n],
                "note": "solve_rate is from the most recently scored checkpoint; gold is never shown"}

    def proxy_stats(self, axis: str, by: str | None) -> dict:
        rows = [{k: v for k, v in r.items() if k != "options"} for r in self._proxy_rows(axis)]
        out = {"axis": axis, "profile": stats.profile(rows)}
        if by:
            out["group_by"] = stats.group_by(rows, by, metric="solve_rate")
        return out

    def _pool_ids(self, in_pool: int | None) -> set[str] | None:
        return None if in_pool is None else {r["id"] for r in read_jsonl(round_dir(self.run, in_pool) / "selection.jsonl")}

    def _select(self, domain, channel, where, in_pool):
        keep, clauses = self._pool_ids(in_pool), hub.parse_where(where)
        for e in self.store.entries():
            if (domain and e["domain"] != domain) or (channel and e["channel"] != channel):
                continue
            if keep is not None and e["id"] not in keep:
                continue
            if clauses and not hub.passes(hub.entry_fields(e), clauses):
                continue
            yield e

    def candidates(self, domain, channel, n, where, in_pool, preview_chars) -> dict:
        pv = max(0, min(preview_chars, 2000))
        matched = list(self._select(domain, channel, where, in_pool))
        return {"n_matching": len(matched), "items": [
            {"id": e["id"], "domain": e["domain"], "channel": e["channel"], "round": e["round"],
             "chars": len(e["instruction"]) + len(e["output"]), "meta": e["meta"],
             "source": e["provenance"].get("repo"),
             "instruction": e["instruction"][:pv], "output": e["output"][:pv]} for e in matched[:n]]}

    def candidates_stats(self, domain, by, in_pool) -> dict:
        rows = [{"domain": e["domain"], "channel": e["channel"], "round": e["round"],
                 "source": e["provenance"].get("repo"), "instruction_chars": len(e["instruction"]),
                 "output_chars": len(e["output"]), **{f"meta.{k}": v for k, v in e["meta"].items()}}
                for e in self._select(domain, None, None, in_pool)]
        out = {"profile": stats.profile(rows)}
        if by:
            out["group_by"] = stats.group_by(rows, by, metric="output_chars")
        return out

    def show(self, cid: str) -> dict:
        e = self.store.get(cid)
        if e is None:
            raise ValueError(f"unknown candidate {cid!r}")
        with self._lock:
            self.views += 1
        return {k: e[k] for k in ("id", "domain", "channel", "instruction", "output", "provenance", "meta")}

    # ------------------------------------------------------------ writes
    def pull(self, b: dict) -> dict:
        if b.get("domain") not in self.cfg["domains"]:
            raise ValueError(f"domain must be one of {list(self.cfg['domains'])}")
        if not (b.get("instruction_field") and b.get("output_field")):
            raise ValueError("pass instruction_field and output_field (see `pb hf schema`)")
        res = self.hub.pull(b["repo"], b.get("config"), b.get("split", "train"), n=int(b["n"]),
                            offset=int(b.get("offset", 0)), instruction_field=b["instruction_field"],
                            output_field=b["output_field"], where=b.get("where"),
                            shuffle_seed=b.get("shuffle_seed"), data_files=b.get("data_files"),
                            scan_limit=int(b.get("scan_limit", 200_000)), round_no=self.t)
        entries = [{"channel": "hf", "domain": b["domain"], **e} for e in res.pop("entries")]
        reg = self.store.register(entries, round_no=self.t, from_gateway=True)
        lengths = sorted(len(e["output"]) for e in entries)
        return {**res, **reg, "output_chars": stats.quantiles(lengths),
                "preview": [{"id": i, "instruction": e["instruction"][:100]}
                            for i, e in zip(reg["ids"], entries[:20])]}

    def submit(self, selection: list, train_config: dict | None) -> dict:
        with self._lock:
            if (self.rd / "SUBMITTED").exists():
                return {"ok": False, "problems": ["this round was already submitted"]}
            plan = self.plan()
            res = submission.audit(self.run, self.cfg, self.store, selection, train_config)
            if not res["ok"]:
                return {k: res[k] for k in ("ok", "problems", "scheduled_tokens", "cap")}
            write_json(self.rd / "plan.json", plan)
            write_jsonl(self.rd / "selection.jsonl", selection)
            write_json(self.rd / "train_config.json", {**res["train_config"], "init_from": res["init_from"]})
            (self.rd / "SUBMITTED").write_text(str(int(time.time())))     # last: marks the others complete
        return {"ok": True, **{k: res[k] for k in ("n_entries", "n_new", "n_kept", "by_domain",
                                                   "scheduled_tokens", "cap", "train_config", "init_from")},
                "note": "submitted; end your session. The harness trains and scores after the round."}


def _handler(g: Gateway):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj, ensure_ascii=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _dispatch(self, fn):
            try:
                self._send(200, fn())
            except (Refused, PermissionError) as e:
                self._send(403, {"error": str(e)})
            except Exception as e:                               # noqa: BLE001
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            one = lambda k, d=None: q.get(k, [d])[0]
            num = lambda k: int(q[k][0]) if k in q else None
            reads = {
                "/proxy/peek": lambda: g.proxy_peek(one("axis"), num("n") or 10, one("wrong") == "1"),
                "/proxy/stats": lambda: g.proxy_stats(one("axis"), one("by")),
                "/candidates": lambda: g.candidates(one("domain"), one("channel"), num("n") or 50,
                                                    q.get("where"), num("in_pool"), num("preview_chars") or 160),
                "/candidates/stats": lambda: g.candidates_stats(one("domain"), one("by"), num("in_pool")),
            }
            free = {"/status": g.status, "/scores": g.scores, "/skill/check": g.skill_check}
            if u.path in free:
                return self._dispatch(free[u.path])
            if u.path in reads:
                return self._dispatch(lambda: (g.gate_read(), reads[u.path]())[1])
            self._send(404, {"error": "no such route"})

        def do_POST(self):
            path = urlparse(self.path).path
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            routes = {
                "/show": lambda: (g.gate_read(), g.show(b["id"]))[1],
                "/hf/search": lambda: (g.gate_remote(), {"results": g.hub.search(b["query"], int(b.get("limit", 20)))})[1],
                "/hf/peek": lambda: (g.gate_read(), g.gate_remote(),
                                     {"rows": g.hub.peek(b["repo"], b.get("config"), b.get("split", "train"),
                                                         int(b.get("n", 3)), b.get("data_files"))})[2],
                "/hf/schema": lambda: (g.gate_read(), g.gate_remote(), stats.profile(
                    g.hub.peek(b["repo"], b.get("config"), b.get("split", "train"), int(b.get("n", 300)),
                               b.get("data_files"), clip=None)))[2],
                "/hf/pull": lambda: (g.plan(), g.gate_remote(), g.pull(b))[2],
                "/register": lambda: (g.plan(), g.store.register(b["entries"], round_no=g.t))[1],
                "/submit": lambda: g.submit(b["selection"], b.get("train_config")),
            }
            if path not in routes:
                return self._send(404, {"error": "no such route"})
            self._dispatch(routes[path])
    return H


def serve(run: Path, cfg: dict, round_no: int, hf_token: str | None):
    """Start the gateway on a free port in a daemon thread; returns (server, gateway, url)."""
    g = Gateway(run, cfg, round_no, hf_token)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(g))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, g, f"http://127.0.0.1:{srv.server_address[1]}"

