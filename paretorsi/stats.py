"""Distributions over rows (a dataset's columns, the store, a proxy set),
without returning row text."""

from __future__ import annotations

from collections import Counter, defaultdict

MAX_CATEGORIES = 50


def quantiles(vals: list) -> dict | None:
    if not vals:
        return None
    v = sorted(vals)
    pick = lambda f: v[min(len(v) - 1, int(f * len(v)))]
    return {"min": v[0], "p10": pick(.1), "p50": pick(.5), "p90": pick(.9), "max": v[-1]}


def profile(rows: list[dict]) -> dict:
    """Per field: categorical counts, numeric quantiles, or text-length quantiles."""
    fields = {}
    for key in sorted({k for r in rows for k in r}):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        info = {"present": len(vals)}
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals):
            info.update(kind="numeric", quantiles=quantiles(vals))
        elif all(isinstance(x, (str, bool)) for x in vals) and len(set(map(str, vals))) <= MAX_CATEGORIES:
            info.update(kind="categorical", counts=dict(Counter(map(str, vals)).most_common(20)))
        elif all(isinstance(x, str) for x in vals):
            info.update(kind="text", chars=quantiles([len(x) for x in vals]))
        else:
            info.update(kind="nested", length=quantiles([len(x) if hasattr(x, "__len__") else 1 for x in vals]))
        fields[key] = info
    return {"n": len(rows), "fields": fields}


def group_by(rows: list[dict], key: str, *, flag: str | None = None, metric: str | None = None) -> dict:
    """Count per value of `key`; with `flag`, the share of rows where it is
    True; with `metric`, that field's median."""
    groups = defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    out = {}
    for value, rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:MAX_CATEGORIES]:
        g = {"n": len(rs)}
        if flag:
            marked = [r[flag] for r in rs if r.get(flag) is not None]
            g[f"{flag}_rate"] = round(sum(marked) / len(marked), 4) if marked else None
        if metric:
            g[f"{metric}_p50"] = (quantiles([r[metric] for r in rs if r.get(metric) is not None]) or {}).get("p50")
        out[value] = g
    return {"by": key, "groups": out}
