"""R_t: the agent's written principles, kept as a Claude Code skill file that it
edits with its ordinary tools.

    workspace/.claude/skills/data-curation/SKILL.md

The file has a harness region (the score history, rewritten every round and
not editable) and slots: one per domain, then Per-channel, Recipe,
Interference, and the optional Evaluation and Allocation, then Retired.  Every
belief is one bullet,

    - [id | basis] text

where basis is `prior` (no evidence yet), evidence such as
`r2:finance+6.3,medical-2.1;r3:code-1.0` (points, per round and axis), or
`retired:rN` under ## Retired.  Cited evidence is checked against the recorded
proxy scores, measured against the parent the round actually trained from.
The budget beta counts only active bullets (chars / 4), so retiring a belief
frees room.  A skill that fails validation at the end of a round is reverted to
the version the harness issued at its start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import models
from .common import read_json

RELPATH = Path(".claude/skills/data-curation/SKILL.md")
REGION = "baseline"
SLOTS = {"per-channel": "channel", "recipe": "recipe", "interference": "interference",
         "evaluation": "evaluation", "allocation": "allocation", "retired": "retired"}
REQUIRED = ("channel", "recipe", "interference")
OPTIONAL = ("evaluation", "allocation")
HINTS = {
    "domain": "What data moves this axis, what does not, and where its ceiling is.",
    "channel": "What each source is worth and what it costs: which repos, which filters, which offsets.",
    "recipe": "What you have learned about lr / epochs / max_len / rank / batch.",
    "interference": "Cross-domain effects: data admitted for one axis moving another.",
    "evaluation": "Observations about the proxy evaluations themselves (noise, coverage, quirks).",
    "allocation": "Your standing allocation intent and why (one or two bullets).",
}

_REGION = re.compile(r"<!-- pb:harness:begin (\w+) -->\n(.*?)\n<!-- pb:harness:end \1 -->", re.S)
_BULLET = re.compile(r"^-\s*\[\s*([A-Za-z0-9_.-]+)\s*\|\s*([^\]]+?)\s*\]\s*(.*)$")
_EVIDENCE = re.compile(r"^r(\d+):(.+)$")
_DELTA = re.compile(r"^([a-z_]+)\s*([+-])\s*(\d*\.?\d+)$", re.I)
_RETIRED = re.compile(r"^retired:r(\d+)$")


@dataclass
class Bullet:
    id: str
    basis: str
    text: str
    slot: str
    line: int

    def evidence(self) -> list[tuple[int, str, float]]:
        out = []
        for group in self.basis.split(";"):
            m = _EVIDENCE.match(group.strip())
            for item in (m.group(2).split(",") if m else []):
                d = _DELTA.match(item.strip())
                if d:
                    out.append((int(m.group(1)), d.group(1).lower(), float(d.group(2) + d.group(3))))
        return out


def path(run: Path) -> Path:
    return Path(run) / "workspace" / RELPATH


def parse(text: str) -> tuple[dict, dict[str, str], list[Bullet]]:
    """(frontmatter, harness regions, bullets with the slot each sits in)."""
    front = {}
    if text.startswith("---\n") and "\n---" in text[4:]:
        head, text = text[4:].split("\n---", 1)
        front = dict(l.split(":", 1) for l in head.splitlines() if ":" in l and not l.startswith(" "))
        front = {k.strip(): v.strip() for k, v in front.items()}
    regions = {m.group(1): m.group(2) for m in _REGION.finditer(text)}
    body = _REGION.sub("", text)
    bullets, slot = [], ""
    for i, line in enumerate(body.splitlines(), 1):
        if line.startswith("## "):
            head = re.sub(r"\s*\(optional\)\s*$", "", line[3:].strip().lower())
            slot = SLOTS.get(head, "domain" if head == "per-domain" else "")
        elif line.startswith("### ") and slot.startswith("domain"):
            slot = f"domain:{line[4:].strip().lower()}"
        elif m := _BULLET.match(line.rstrip()):
            bullets.append(Bullet(m.group(1), m.group(2).strip(), m.group(3), slot, i))
        elif bullets and line.startswith(("  ", "\t")) and line.strip():
            bullets[-1].text += " " + line.strip()
    return front, regions, bullets


def score_history(run: Path) -> dict[int, dict[str, float]]:
    """{round: {domain: proxy score}}, round 0 being M_0."""
    out = {}
    for f in sorted((Path(run) / "scores").glob("round_*.json")):
        s = read_json(f)
        out[s["round"]] = dict(zip(s["order"], s["vector"]))
    return out


def parents(run: Path) -> dict[int, int]:
    return {h["round"]: (0 if h["init_from"] == "base" else int(h["init_from"].split("_")[1]))
            for h in models.history(run) if h["status"] == "ok"}


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def validate(text: str, *, cfg: dict, round_no: int, issued_regions: dict[str, str],
             history: dict, parent_of: dict[int, int]) -> Report:
    r = Report()
    doms = list(cfg["domains"])
    front, regions, bullets = parse(text)
    for k in ("name", "description"):
        if not front.get(k):
            r.problems.append(f"frontmatter is missing {k!r}")
    if {k: v.strip() for k, v in regions.items()} != {k: v.strip() for k, v in issued_regions.items()}:
        r.problems.append("the harness region was edited, removed or added to; it is harness-owned")

    seen = {}
    for b in bullets:
        if b.id in seen:
            r.problems.append(f"duplicate id {b.id!r} (lines {seen[b.id]} and {b.line})")
        seen[b.id] = b.line
        if not b.slot or b.slot == "domain" or (b.slot.startswith("domain:") and b.slot[7:] not in doms):
            r.problems.append(f"{b.id!r} (line {b.line}) is not under a slot heading of the template")
        if not b.text.strip():
            r.problems.append(f"{b.id!r} has no text")
    active = [b for b in bullets if b.slot != "retired"]
    filled = {b.slot for b in active}
    required = [f"domain:{d}" for d in doms] + [s for s in REQUIRED if s != "interference" or len(doms) > 1]
    missing = [s for s in required if s not in filled]
    if missing:
        r.problems.append(f"empty required slots {missing}: write at least one bullet in each (`prior` is fine)")

    tol = float(cfg["principles"]["cite_tolerance_pp"])
    for b in bullets:
        m = _RETIRED.match(b.basis)
        if (b.slot == "retired") != bool(m):
            r.problems.append(f"{b.id!r}: `retired:rN` belongs under ## Retired and only there")
            continue
        if m:
            if int(m.group(1)) > round_no:
                r.problems.append(f"{b.id!r} is retired in round {m.group(1)}, which has not happened")
            continue
        if b.basis == "prior":
            continue
        ev = b.evidence()
        if not ev:
            r.problems.append(f"{b.id!r}: basis {b.basis!r} is not `prior`, `rN:<axis>±<points>,...` or `retired:rN`")
        for rnd, dom, claimed in ev:
            if rnd not in history or rnd >= round_no:
                r.problems.append(f"{b.id!r} cites round {rnd}, which has no recorded score")
                continue
            par = parent_of.get(rnd, rnd - 1)
            if dom not in history[rnd] or dom not in history.get(par, {}):
                r.problems.append(f"{b.id!r} cites unknown axis {dom!r}")
                continue
            actual = 100 * (history[rnd][dom] - history[par][dom])
            if abs(actual - claimed) > tol:
                r.problems.append(f"{b.id!r} claims {dom}{claimed:+.1f} in round {rnd}; the record says "
                                  f"{actual:+.1f} (against its parent, {'M_0' if par == 0 else f'r{par}'})")

    tokens = (len("\n".join(f"- [{b.id} | {b.basis}] {b.text}" for b in active)) + 3) // 4
    beta = int(cfg["principles"]["max_tokens"])
    if tokens > beta:
        r.problems.append(f"active bullets are {tokens} tokens; the budget is {beta}. Retire or shorten")
    r.stats = {"round": round_no, "n_active": len(active), "n_retired": len(bullets) - len(active),
               "n_prior": sum(b.basis == "prior" for b in active), "tokens": tokens, "beta": beta}
    return r


# ------------------------------------------------------------------ rendering

def render_region(cfg: dict, history: dict, parent_of: dict[int, int]) -> str:
    rounds = sorted(history)
    lines = ["## Baseline and history", "",
             "| axis | " + " | ".join("M_0" if t == 0 else f"r{t}" for t in rounds) + " |",
             "|---|" + "---|" * len(rounds)]
    for d in cfg["domains"]:
        lines.append(f"| {d} | " + " | ".join(
            f"{history[t][d]:.3f}" if history[t].get(d) is not None else "-" for t in rounds) + " |")
    branched = {t: p for t, p in parent_of.items() if p != t - 1}
    if branched:
        lines += ["", "Rounds not trained from their predecessor: " + ", ".join(
            f"r{t} from {'M_0' if p == 0 else f'r{p}'}" for t, p in sorted(branched.items()))]
    if len(rounds) > 1:
        t = rounds[-1]
        p = parent_of.get(t, t - 1)
        lines += ["", f"Change in round {t} against its parent ({'M_0' if p == 0 else f'r{p}'}), in points: "
                  + "  ".join(f"{d}{100 * (history[t][d] - history[p][d]):+.1f}" for d in cfg["domains"])]
    return "\n".join(lines)


def _wrap(body: str) -> str:
    return f"<!-- pb:harness:begin {REGION} -->\n{body}\n<!-- pb:harness:end {REGION} -->"


def initial(cfg: dict, history: dict) -> str:
    doms = list(cfg["domains"])
    ex = doms[:2] if len(doms) > 1 else doms * 2
    slots = ["## Per-domain", ""]
    for d in doms:
        slots += [f"### {d}", f"<!-- {HINTS['domain']} -->", ""]
    for head, key in (("Per-channel", "channel"), ("Recipe", "recipe"), ("Interference", "interference"),
                      ("Evaluation", "evaluation"), ("Allocation", "allocation")):
        if key in ("interference", "allocation") and len(doms) < 2:
            continue
        slots += [f"## {head}{' (optional)' if key in OPTIONAL else ''}", f"<!-- {HINTS[key]} -->", ""]
    slots += ["## Retired", ""]
    return f"""---
name: data-curation
description: Accumulated principles for curating post-training data across {', '.join(doms)}. Read this at the start of every round, before deciding anything, and revise it at the end of the round.
---

{_wrap(render_region(cfg, history, {}))}

## How to maintain this file

Your scratch directory round_NN/ is archived when the round ends; your own
conversation carries over, until the context window compacts it. This file is
what you deliberately keep.

Every slot below is one dimension of the decision you make each round. Fill
each required slot with at least one bullet; in round 1 that is a belief held
before any evidence, tagged `prior`.

Every bullet is tagged `- [id | basis] text`:

- `prior`: you believe it, but no round has shown it yet.
- `r2:{ex[0]}+6.3,{ex[1]}-2.1`: round 2 moved those axes by those points, against
  the checkpoint it trained from. The harness checks the numbers against the
  recorded scores; a skill with a wrong citation is reverted at round end.
- `retired:r4`: you gave up on it in round 4. Move it under `## Retired` and
  say what falsified it.

The budget is {cfg['principles']['max_tokens']} tokens (characters / 4) of active bullets; retiring
a belief frees its room. New bullets go under the existing headings only.

""" + "\n".join(slots)


def refresh(run: Path, cfg: dict) -> None:
    """Rewrite the harness region in place, leaving everything else as the agent wrote it."""
    p = path(run)
    region = _wrap(render_region(cfg, score_history(run), parents(run)))
    new, n = _REGION.subn(lambda _: region, p.read_text(), count=1)
    if n != 1:
        raise RuntimeError(f"{p}: harness region not found")
    p.write_text(new)
