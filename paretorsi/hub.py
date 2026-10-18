"""Search, Select and Pull on the Hugging Face Hub, done by the harness on the
agent's behalf.  Rows a pull returns go straight into the candidate store; the
agent never handles their text on the way in, so the provenance the gateway
writes ({repo, config, split, row}) is the verification.

The sources of every benchmark this run evaluates on are refused at the
source (BLOCKED_*).  This catches the obvious cases early and with a reason;
n-gram decontamination at registration is the guard that does not depend on
anyone keeping a list up to date.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .common import read_jsonl

# The upstream repositories of the ten benchmark files, and their parents:
# MATH-500 is the test split of Hendrycks MATH; MBPP+ and HumanEval+ are
# re-tested versions of MBPP and HumanEval.
BLOCKED_REPOS = {
    "HuggingFaceH4/MATH-500", "hendrycks/competition_math", "EleutherAI/hendrycks_math",
    "lighteval/MATH", "lighteval/MATH-Hard", "HuggingFaceTB/MATH", "qq8933/MATH500",
    "nlile/hendrycks-MATH-benchmark", "Idavidrein/gpqa", "LangAGI-Lab/medbullets_op5",
    "LangAGI-Lab/medbullets", "HanjieChen/ChallengeClinicalQA", "m-a-p/SuperGPQA",
    "evalplus/mbppplus", "evalplus/humanevalplus", "google-research-datasets/mbpp",
    "openai/openai_humaneval",
}
# Mirrors rarely keep the repo name but usually keep the benchmark's.
BLOCKED_SUBSTRINGS = ("math-500", "math500", "hendrycks", "gpqa", "medbullets",
                      "challengeclinicalqa", "mbpp", "humaneval", "human-eval", "paretorsi")


def blocked(repo: str) -> str | None:
    if repo in BLOCKED_REPOS:
        return f"{repo} is the source of an evaluation benchmark"
    low = repo.lower()
    for s in BLOCKED_SUBSTRINGS:
        if s in low:
            return f"{repo} looks like a copy of an evaluation benchmark (name contains {s!r})"
    return None


# --------------------------------------------------------------- row filters

_OPS = {
    "==": lambda a, b: str(a) == b,
    "!=": lambda a, b: str(a) != b,
    ">=": lambda a, b: _num(a) is not None and _num(a) >= float(b),
    "<=": lambda a, b: _num(a) is not None and _num(a) <= float(b),
    ">": lambda a, b: _num(a) is not None and _num(a) > float(b),
    "<": lambda a, b: _num(a) is not None and _num(a) < float(b),
    "!~": lambda a, b: b.lower() not in str(a).lower(),
    "~": lambda a, b: b.lower() in str(a).lower(),
}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_where(clauses: list[str] | None) -> list[tuple[str, str, str]]:
    """`field<op>value`, one comparison per clause; clauses are AND-ed."""
    out = []
    for c in clauses or []:
        if re.search(r"\s(and|or|&&|\|\|)\s", c, re.I):
            raise ValueError(f"--where {c!r}: one comparison per clause; repeat --where to AND them")
        op = next((o for o in ("==", "!=", ">=", "<=", "!~", ">", "<", "~") if o in c), None)
        if op is None:
            raise ValueError(f"--where {c!r}: expected <field><op><value>, op in {list(_OPS)}")
        field, value = c.split(op, 1)
        out.append((field.strip(), op, value.strip()))
    return out


def passes(fields: dict, where) -> bool:
    return all(f in fields and _OPS[op](fields[f], v) for f, op, v in where)


def field_path(row, path: str):
    """`answer`, `messages[1].content`, `conversations[0].value`; None if absent."""
    cur = row
    for name, idx in re.findall(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]", path):
        try:
            cur = cur[int(idx)] if idx else cur[name]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def map_row(row: dict, instruction_field: str, output_field: str) -> dict | None:
    """The row as a training entry, with its other short scalar columns as meta."""
    ins, out = field_path(row, instruction_field), field_path(row, output_field)
    if not ins or not out:
        return None
    top = {re.split(r"[.\[]", instruction_field)[0], re.split(r"[.\[]", output_field)[0]}
    meta = {k: v for k, v in row.items()
            if k not in top and isinstance(v, (str, int, float, bool)) and len(str(v)) < 200}
    return {"instruction": str(ins), "output": str(out), "meta": meta}


def entry_fields(e: dict) -> dict:
    """The namespace a --where clause sees on a mapped entry."""
    return {"instruction": e["instruction"], "output": e["output"],
            "instruction_chars": len(e["instruction"]), "output_chars": len(e["output"]),
            **{f"meta.{k}": v for k, v in (e.get("meta") or {}).items()}}


# ----------------------------------------------------------------------- hub

class Hub:
    def __init__(self, run: Path, *, byte_cap: int, token: str | None):
        self.log_path = Path(run) / "store" / "hf_log.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.byte_cap, self.token = byte_cap, token
        self.bytes_used = sum(r["bytes"] for r in read_jsonl(self.log_path))
        self._sizes: dict[str, int | None] = {}
        self._lock = threading.Lock()

    def _stream(self, repo, config, split, data_files=None):
        from datasets import load_dataset
        return load_dataset(repo, config, split=split, streaming=True, token=self.token,
                            data_files=data_files)

    def search(self, query: str, limit: int = 20) -> list[dict]:
        from huggingface_hub import HfApi
        found = HfApi(token=self.token).list_datasets(search=query, limit=limit, sort="downloads")
        return [{"repo": d.id, "downloads": d.downloads, "blocked": blocked(d.id)} for d in found]

    def peek(self, repo, config, split, n=3, data_files=None, clip: int | None = 400) -> list[dict]:
        if reason := blocked(repo):
            raise PermissionError(reason)
        rows = []
        for r in self._stream(repo, config, split, data_files):
            rows.append({k: (v if clip is None or len(str(v)) <= clip else str(v)[:clip] + "...")
                         for k, v in r.items()})
            if len(rows) >= n:
                break
        return rows

    def rows_upstream(self, repo, config, split) -> int | None:
        """Rows in the split, from the datasets-server; None when unknown."""
        key = f"{repo}|{config}|{split}"
        if key not in self._sizes:
            try:
                url = "https://datasets-server.huggingface.co/size?dataset=" + urllib.parse.quote(repo, safe="")
                req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.token}"} if self.token else {})
                with urllib.request.urlopen(req, timeout=20) as r:
                    splits = json.load(r)["size"]["splits"]
                match = [s for s in splits if s["split"] == split and (config is None or s["config"] == config)]
                self._sizes[key] = sum(s["num_rows"] for s in match[:1]) if match else None
            except Exception:                                    # noqa: BLE001
                self._sizes[key] = None
        return self._sizes[key]

    def pull(self, repo, config, split, *, n, offset, instruction_field, output_field,
             where=None, shuffle_seed=None, data_files=None, scan_limit=200_000, round_no: int) -> dict:
        """Scan the split from `offset`, map each row with the two fields, keep
        those passing `where`, stop at n kept rows (or the scan limit, or the
        run's byte cap).  `next_offset` continues past everything scanned."""
        if reason := blocked(repo):
            raise PermissionError(reason)
        clauses = parse_where(where)
        ds = self._stream(repo, config, split, data_files)
        if shuffle_seed is not None:
            ds = ds.shuffle(seed=shuffle_seed, buffer_size=1000)
        kept, scanned, unmapped, filtered, nbytes = [], 0, 0, 0, 0
        for i, row in enumerate(ds.skip(offset)):
            if scanned >= scan_limit or len(kept) >= n:
                break
            scanned += 1
            e = map_row(row, instruction_field, output_field)
            if e is None:
                unmapped += 1
                continue
            if clauses and not passes(entry_fields(e), clauses):
                filtered += 1
                continue
            size = len(json.dumps(e, ensure_ascii=False, default=str))
            if self.bytes_used + nbytes + size > self.byte_cap:
                break
            nbytes += size
            pos = offset + i
            e["provenance"] = {"repo": repo, "config": config, "split": split,
                               "row": f"s{shuffle_seed}:{pos}" if shuffle_seed is not None else pos}
            kept.append(e)
        rec = {"ts": int(time.time()), "round": round_no, "repo": repo, "config": config, "split": split,
               "fields": [instruction_field, output_field], "where": where or [],
               "shuffle_seed": shuffle_seed, "offset": offset, "next_offset": offset + scanned,
               "scanned": scanned, "kept": len(kept), "unmapped": unmapped, "filtered": filtered,
               "bytes": nbytes}
        with self._lock:
            self.bytes_used += nbytes
            with self.log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return {**rec, "entries": kept, "hf_bytes_used": self.bytes_used, "hf_byte_cap": self.byte_cap,
                "rows_upstream": self.rows_upstream(repo, config, split)}

    def supply(self) -> list[dict]:
        """Per pulled split: rows scanned so far by this run vs rows upstream."""
        seen: dict[tuple, int] = {}
        for r in read_jsonl(self.log_path):
            k = (r["repo"], r["config"], r["split"])
            seen[k] = max(seen.get(k, 0), r["next_offset"]) if r["shuffle_seed"] is None else seen.get(k, 0)
        return [{"repo": repo, "config": config, "split": split, "scanned_up_to": off,
                 "rows_upstream": self.rows_upstream(repo, config, split)}
                for (repo, config, split), off in seen.items()]
