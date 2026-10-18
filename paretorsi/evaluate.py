"""Score one model on the proxy or held-out suite.

    python -m paretorsi.evaluate --config <run>/config.lock.yaml --suite proxy \
        --model <run>/round_03/train/model --out <run>/round_03/eval

Sampling runs in one vLLM engine with the decoding block of the config and
nothing else: no sampling parameter is a flag, so a score cannot be improved
by changing how it was sampled.  Prompts are token ids from `render`, the same
ids training used.  One `generate` call per domain, items in id order, so the
batch a domain is sampled in does not depend on any other domain.

Per domain, under <out>/<domain>/:
    samples.jsonl   the k responses per item (never the gold)
    per_item.json   {id: share of the k samples judged correct}
    report.json     every metric, the headline, extraction and truncation stats
and <out>/eval.json with the headline vector.  A domain with a report.json is
not sampled again, so an interrupted evaluation resumes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import render, tasks
from .common import domains, load_benchmark, load_config, read_json, write_json, write_jsonl


def _engine(model: str, tokenizer: str, max_model_len: int, seed: int):
    from transformers import AutoConfig
    from vllm import LLM
    kw = {}
    if AutoConfig.from_pretrained(model).model_type == "mistral3":
        kw["limit_mm_per_prompt"] = {"image": 0}      # text-only use of a vision-language checkpoint
    return LLM(model=model, tokenizer=tokenizer, dtype="bfloat16", seed=seed,
               max_model_len=max_model_len, gpu_memory_utilization=0.9, **kw)


def evaluate(cfg: dict, suite: str, model: str, out: Path, only: list[str] | None = None) -> dict:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    dec = cfg["decoding"]
    k = int(dec["k"])
    tok = render.load_tokenizer(cfg["base_model"])
    todo = [d for d in (only or domains(cfg)) if not (out / d / "report.json").exists()]
    work = {}
    for dom in todo:
        ev = cfg["domains"][dom][suite]
        items = load_benchmark(ev)
        work[dom] = (ev, items, [render.prompt_ids(tok, tasks.prompt(ev["task"], it)) for it in items])
    if work:
        longest = max(len(ids) for _, _, prompts in work.values() for ids in prompts)
        llm = _engine(model, cfg["base_model"], longest + int(dec["max_tokens"]), int(dec["seed"]))
        params = SamplingParams(n=k, temperature=dec["temperature"], top_p=dec["top_p"],
                                top_k=dec["top_k"], seed=int(dec["seed"]),
                                max_tokens=int(dec["max_tokens"]),
                                stop_token_ids=render.stop_ids(tok), stop=render.stop_strings(tok),
                                skip_special_tokens=True)
        for dom, (ev, items, prompts) in work.items():
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], params)
            rows = [{"id": it["id"], "prompt_tokens": len(ids),
                     "samples": [{"text": c.text, "n_tokens": len(c.token_ids),
                                  "finish_reason": c.finish_reason} for c in o.outputs]}
                    for it, ids, o in zip(items, prompts, outs)]
            _grade(dom, ev, items, rows, k, out / dom)

    vector = []
    for dom in domains(cfg):
        rep = read_json(out / dom / "report.json")
        vector.append(rep["score"] if rep else None)
    scored = [v for v in vector if v is not None]
    result = {"suite": suite, "model": model, "order": domains(cfg), "vector": vector,
              "mean": round(sum(scored) / len(scored), 4) if scored else None}
    write_json(out / "eval.json", result)
    return result


def _grade(dom: str, ev: dict, items: list[dict], rows: list[dict], k: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "samples.jsonl", rows)
    task = ev["task"]
    by_id = {it["id"]: it for it in items}
    jobs = [(r["id"], s["text"]) for r in rows for s in r["samples"]]
    if task == "code":
        with ThreadPoolExecutor(max_workers=16) as pool:   # each judgement is a subprocess
            results = list(pool.map(lambda j: tasks.judge(task, by_id[j[0]], j[1]), jobs))
    else:
        results = [tasks.judge(task, by_id[i], text) for i, text in jobs]
    verdicts, answers, how = {}, {}, Counter()
    for (iid, _), (ok, ans, tag) in zip(jobs, results):
        verdicts.setdefault(iid, []).append(bool(ok))
        answers.setdefault(iid, []).append(ans)
        how[tag] += 1
    scores = tasks.score(task, items, verdicts, answers, k)
    samples = [s for r in rows for s in r["samples"]]
    write_json(out / "per_item.json", {i: round(sum(v) / k, 4) for i, v in verdicts.items()})
    write_json(out / "report.json", {
        "domain": dom, "task": task, "benchmark": Path(ev["data"]).name, "n": len(items), "k": k,
        "metric": ev["metric"], "score": scores[f"{ev['metric']}@{k}"], "scores": scores,
        "extraction": {t: round(c / len(samples), 4) for t, c in sorted(how.items())},
        "truncation_rate": round(sum(s["finish_reason"] == "length" for s in samples) / len(samples), 4),
        "mean_output_tokens": round(sum(s["n_tokens"] for s in samples) / len(samples), 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", choices=["proxy", "heldout"], required=True)
    ap.add_argument("--model", required=True, help="model directory or hub id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default=None, help="comma-separated subset")
    a = ap.parse_args()
    cfg = load_config(Path(a.config))
    res = evaluate(cfg, a.suite, a.model, Path(a.out), a.domains.split(",") if a.domains else None)
    print(res)


if __name__ == "__main__":
    main()
