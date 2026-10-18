"""The candidate store: the only way data reaches a training pool.

Every entry is registered here.  The harness assigns the id, hashes the text,
writes the provenance and runs decontamination; the agent's selection may only
name ids that exist here.  Two channels:

    hf       a row the gateway pulled from a Hugging Face dataset.
             provenance {repo, config, split, row}, written by the gateway.
    derived  a rewrite the agent made of a stored entry.
             provenance {parent_id, transform}; the parent must exist and the
             text must differ from it.

Decontamination: an entry whose instruction+output shares any informative
13-gram (lower-cased word tokens) with any proxy or held-out item is rejected
at registration, with the reason, so contaminated text is never selectable.

Files, append-only:  <run>/store/candidates.jsonl, <run>/store/rejected.jsonl
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from .common import SUITES, load_benchmark, read_jsonl

CHANNELS = ("hf", "derived")
MIN_OUTPUT_CHARS = 16
_WORD = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------- decontamination

def ngrams(text: str, n: int):
    """Word n-grams that carry information.  Runs of option letters or numbers
    ("a b c d a b c d ...") occur in every multiple-choice item and in ordinary
    prose alike; matching them would be a false positive, not a leak."""
    toks = _WORD.findall(text.lower())
    for i in range(len(toks) - n + 1):
        g = toks[i:i + n]
        if len(set(g)) >= 6 and sum(map(len, g)) >= 30:
            yield " ".join(g)


def item_text(item: dict) -> str:
    """All of an evaluation item a training entry must not overlap: the
    question, its options and its gold, including a code item's tests."""
    parts = [item.get(k) for k in ("problem", "question", "prompt", "answer", "label", "test")]
    opts = item.get("options")
    parts += list(opts.values()) if isinstance(opts, dict) else []
    parts += list(item.get("assertions") or [])
    return " ".join(p for p in parts if isinstance(p, str))


def eval_ngrams(cfg: dict) -> set[str]:
    n = int(cfg["decontamination"]["ngram"])
    grams: set[str] = set()
    for spec in cfg["domains"].values():
        for suite in SUITES:
            for item in load_benchmark(spec[suite]):
                grams.update(ngrams(item_text(item), n))
    return grams


def first_hit(text: str, grams: set[str], n: int) -> str | None:
    return next((g for g in ngrams(text, n) if g in grams), None)


_GRAMS: set[str] = set()
_N = 13


def _init_worker(grams, n):
    global _GRAMS, _N
    _GRAMS, _N = grams, n


def _hits(texts):
    return [first_hit(t, _GRAMS, _N) for t in texts]


# ----------------------------------------------------------------- the store

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CandidateStore:
    def __init__(self, run: Path, cfg: dict):
        self.dir = Path(run) / "store"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "candidates.jsonl"
        self.rejected_path = self.dir / "rejected.jsonl"
        self.domains = list(cfg["domains"])
        self.n = int(cfg["decontamination"]["ngram"])
        self.grams = eval_ngrams(cfg)
        self._lock = threading.Lock()
        self._pool = None
        self._by_id: dict[str, dict] = {}
        self._by_sha: dict[str, str] = {}
        self._seq = {c: 0 for c in CHANNELS}
        for e in read_jsonl(self.path):
            self._index(e)

    def _index(self, e: dict) -> None:
        self._by_id[e["id"]] = e
        self._by_sha[e["sha256"]] = e["id"]
        _, ch, num = e["id"].split(":")
        self._seq[ch] = max(self._seq[ch], int(num))

    def get(self, cid: str) -> dict | None:
        return self._by_id.get(cid)

    def has(self, cid) -> bool:
        return isinstance(cid, str) and cid in self._by_id

    def entries(self):
        return self._by_id.values()

    def summary(self) -> dict:
        by = lambda key: {v: sum(1 for e in self._by_id.values() if e[key] == v)
                          for v in sorted({e[key] for e in self._by_id.values()})}
        return {"n": len(self._by_id), "by_domain": by("domain"), "by_channel": by("channel")}

    def _contamination(self, texts: list[str]) -> list[str | None]:
        if len(texts) < 64:
            return [first_hit(t, self.grams, self.n) for t in texts]
        if self._pool is None:
            # forked workers inherit the n-gram set instead of pickling it per task
            self._pool = ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1),
                                             mp_context=get_context("fork"),
                                             initializer=_init_worker, initargs=(self.grams, self.n))
        size = max(16, len(texts) // 32)
        chunks = [texts[i:i + size] for i in range(0, len(texts), size)]
        return [h for part in self._pool.map(_hits, chunks) for h in part]

    def _reject_reason(self, e: dict, hit: str | None, from_gateway: bool) -> str | None:
        # `meta` is the source row's own columns, attached by the gateway only
        allowed = {"channel", "domain", "instruction", "output", "provenance"} | ({"meta"} if from_gateway else set())
        if set(e) - allowed:
            return f"unexpected fields {sorted(set(e) - allowed)}"
        if e.get("channel") not in CHANNELS:
            return f"channel must be one of {CHANNELS}"
        if e["channel"] == "hf" and not from_gateway:
            return "hf rows enter the store when `pb hf pull` fetches them; your own text registers as 'derived'"
        if e.get("domain") not in self.domains:
            return f"domain must be one of {self.domains}"
        ins, out = e.get("instruction"), e.get("output")
        if not isinstance(ins, str) or not ins.strip():
            return "instruction is empty"
        if not isinstance(out, str) or len(out.strip()) < MIN_OUTPUT_CHARS:
            return f"output is shorter than {MIN_OUTPUT_CHARS} characters"
        if e["channel"] == "derived":
            prov = e.get("provenance") or {}
            parent = self._by_id.get(prov.get("parent_id"))
            if parent is None:
                return "derived entries need provenance.parent_id naming a stored entry"
            if not str(prov.get("transform", "")).strip():
                return "derived entries need provenance.transform describing the rewrite"
            if (parent["instruction"], parent["output"]) == (ins, out):
                return "derived entry is identical to its parent"
        if hit:
            return f"contaminated: shares the {self.n}-gram {hit[:80]!r} with an evaluation item"
        return None

    def register(self, entries: list[dict], *, round_no: int, from_gateway: bool = False) -> dict:
        """Accept or reject each entry.  An entry whose text is already stored
        returns the existing id, so registering is idempotent."""
        hits = self._contamination([f"{e.get('instruction', '')} {e.get('output', '')}"
                                    if isinstance(e, dict) else "" for e in entries])
        accepted, rejected, duplicates = [], [], 0
        with self._lock, self.path.open("a", encoding="utf-8") as fa, \
                self.rejected_path.open("a", encoding="utf-8") as fr:
            for i, (e, hit) in enumerate(zip(entries, hits)):
                reason = (self._reject_reason(e, hit, from_gateway) if isinstance(e, dict)
                          else "entry is not an object")
                if reason:
                    rejected.append({"index": i, "reason": reason})
                    fr.write(json.dumps({"round": round_no, "index": i, "reason": reason,
                                         "channel": e.get("channel") if isinstance(e, dict) else None,
                                         "ts": int(time.time())}) + "\n")
                    continue
                h = sha(e["instruction"] + "\n" + e["output"])
                if h in self._by_sha:
                    accepted.append(self._by_sha[h])
                    duplicates += 1
                    continue
                self._seq[e["channel"]] += 1
                rec = {"id": f"cand:{e['channel']}:{self._seq[e['channel']]:07d}",
                       "channel": e["channel"], "domain": e["domain"],
                       "instruction": e["instruction"], "output": e["output"], "sha256": h,
                       "provenance": e.get("provenance") or {}, "meta": e.get("meta") or {},
                       "round": round_no, "ts": int(time.time())}
                fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._index(rec)
                accepted.append(rec["id"])
            fa.flush()
        return {"ids": accepted, "n_new": len(accepted) - duplicates, "n_duplicate": duplicates,
                "n_rejected": len(rejected), "rejected": rejected[:50]}

    def materialize(self, ids: list[str]) -> list[dict]:
        return [{k: self._by_id[c][k] for k in ("id", "domain", "channel", "instruction", "output")}
                for c in ids]
