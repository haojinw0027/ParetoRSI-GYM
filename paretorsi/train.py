"""One round of LoRA SFT: one pass over the pool, loss on completion tokens only.

    torchrun --standalone --nproc_per_node=<gpus> -m paretorsi.train \
        --base <model> --data pool.jsonl --out <round>/train [--init-from <state dir>] \
        --lr 1e-5 --lora-rank 32 --batch-size 64 --epochs 1 --max-len 8192 --warmup-steps 10 --seed 0

A step is one batch of --batch-size examples across all ranks.  Each rank
takes every world-th example of the batch, runs it in micro-batches of at most
MICRO_BATCH_TOKENS padded tokens, and sums the per-token loss; the LoRA
gradients and token counts are then all-reduced, so the update is the gradient
of the mean loss over every completion token in the batch, whatever the
number of GPUs.  LR: linear warmup, then cosine decay to 10%.

Writes to --out:
    state/          LoRA adapter + optimizer.pt; a later round resumes from it
    model/          base + adapter merged, what evaluation loads
    train_log.jsonl per-step loss
    report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

from . import render
from .common import read_jsonl

MICRO_BATCH_TOKENS = 16384
LORA_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_base(model: str, dtype):
    """Ministral-3 ships as a vision-language checkpoint (model_type mistral3);
    only its language model is trained."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(model)
    multimodal = config.model_type == "mistral3"
    cls = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    base = cls.from_pretrained(model, dtype=dtype, attn_implementation="sdpa")
    targets = (r"model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(" + "|".join(LORA_MODULES) + ")"
               if multimodal else list(LORA_MODULES))
    return base, targets, multimodal


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = (step - warmup) / max(total - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def micro_batches(lengths: dict[int, int], idxs: list[int]):
    """Group length-sorted examples so that rows x longest row <= MICRO_BATCH_TOKENS."""
    cur, longest = [], 0
    for i in sorted(idxs, key=lambda i: lengths[i]):
        m = max(longest, lengths[i])
        if cur and (len(cur) + 1) * m > MICRO_BATCH_TOKENS:
            yield cur
            cur, m = [], lengths[i]
        cur.append(i)
        longest = m
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", default=None, help="state/ directory of the parent round")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lora-rank", type=int, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--warmup-steps", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, PeftModel, get_peft_model

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    main_rank = rank == 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    tok = render.load_tokenizer(a.base)
    rows = read_jsonl(Path(a.data))
    examples = []                        # (ids, number of prompt tokens)
    for r in rows:
        p, c = render.prompt_ids(tok, r["instruction"]), render.target_ids(tok, r["output"])
        ids = (p + c)[:a.max_len]
        if len(ids) > len(p):            # at least one completion token survives truncation
            examples.append((ids, len(p)))
    lengths = {i: len(ids) for i, (ids, _) in enumerate(examples)}

    base, targets, multimodal = load_base(a.base, torch.bfloat16)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.init_from:
        model = PeftModel.from_pretrained(base, a.init_from, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=a.lora_rank, lora_alpha=2 * a.lora_rank, lora_dropout=0.0,
                                                target_modules=targets, task_type="CAUSAL_LM"))
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    if a.init_from and (Path(a.init_from) / "optimizer.pt").exists():
        opt.load_state_dict(torch.load(Path(a.init_from) / "optimizer.pt", map_location=device))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def loss_sum(mb):
        width = max(lengths[i] for i in mb)
        ids = torch.full((len(mb), width), pad, dtype=torch.long)
        labels = torch.full((len(mb), width), -100, dtype=torch.long)
        mask = torch.zeros((len(mb), width), dtype=torch.long)
        for row, i in enumerate(mb):
            seq, n_prompt = examples[i]
            ids[row, :len(seq)] = torch.tensor(seq)
            labels[row, n_prompt:len(seq)] = torch.tensor(seq[n_prompt:])
            mask[row, :len(seq)] = 1
        ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        target = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                                 ignore_index=-100, reduction="sum")
        return loss, int((target != -100).sum())

    order_rng = random.Random(a.seed)
    steps_per_epoch = math.ceil(len(examples) / a.batch_size)
    total = steps_per_epoch * a.epochs
    log = (out / "train_log.jsonl").open("w") if main_rank else None
    t0, step = time.time(), 0
    for epoch in range(a.epochs):
        order = list(range(len(examples)))
        order_rng.shuffle(order)
        for b in range(0, len(order), a.batch_size):
            lr = lr_at(step, total, a.lr, a.warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            local_loss, local_tokens = 0.0, 0
            for mb in micro_batches(lengths, order[b:b + a.batch_size][rank::world]):
                loss, n = loss_sum(mb)
                loss.backward()
                local_loss += float(loss.detach())
                local_tokens += n
            # sum gradients and token counts over ranks, then divide by the
            # global token count: the gradient of the batch's mean token loss
            totals = torch.tensor([local_loss, local_tokens], device=device, dtype=torch.float64)
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            if world > 1:
                dist.all_reduce(totals)
                for g in grads:
                    dist.all_reduce(g)
            n_tokens = max(float(totals[1]), 1.0)
            for p, g in zip(params, grads):
                p.grad = g / n_tokens
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if main_rank:
                log.write(json.dumps({"step": step, "epoch": epoch, "lr": lr,
                                      "loss": round(float(totals[0]) / n_tokens, 5),
                                      "tokens": int(totals[1]), "elapsed_s": round(time.time() - t0, 1)}) + "\n")
                log.flush()

    if main_rank:
        log.close()
        state = out / "state"
        model.save_pretrained(state)
        torch.save(opt.state_dict(), state / "optimizer.pt")
        merged = model.merge_and_unload()
        # transformers 5 would otherwise rename mistral3 weights back to a
        # legacy layout that vLLM does not load
        merged.save_pretrained(out / "model", **({"save_original_format": False} if multimodal else {}))
        if multimodal:
            _copy_processor_files(a.base, out / "model")
        json.dump({"base": a.base, "init_from": a.init_from, "n_examples": len(examples),
                   "n_dropped": len(rows) - len(examples),
                   "train_tokens": sum(lengths.values()) * a.epochs, "steps": step,
                   "world_size": world, "wall_clock_s": round(time.time() - t0, 1),
                   "recipe": {"lr": a.lr, "lora_rank": a.lora_rank, "batch_size": a.batch_size,
                              "epochs": a.epochs, "max_len": a.max_len,
                              "warmup_steps": a.warmup_steps, "seed": a.seed}},
                  (out / "report.json").open("w"), indent=2)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _copy_processor_files(base: str, dst: Path) -> None:
    """vLLM reads the processor config of a vision-language architecture even
    when images are disabled."""
    from huggingface_hub import snapshot_download
    src = Path(base) if Path(base).is_dir() else Path(snapshot_download(base, allow_patterns=["*.json"]))
    for name in ("processor_config.json", "preprocessor_config.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


if __name__ == "__main__":
    main()
