"""Token-level prompt and target encoding, shared by training, the budget check
and evaluation so all three see identical token ids.

A model with a chat template is prompted through it.  A model without one
(Olmo-3 and Ministral-3 base) is prompted with ChatML written as plain text.

The training target is `output` followed by the tokenizer's EOS id.  Generation
stops on EOS, on every turn marker that is a single token, and, for ChatML
written as text, on the markers as strings, so a base model that writes
"<|im_end|>" and continues is cut there instead of having its continuation
graded.
"""

from __future__ import annotations

from functools import lru_cache

CHATML = "<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
TURN_MARKERS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>")


@lru_cache(maxsize=4)
def load_tokenizer(model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def has_template(tok) -> bool:
    return bool(getattr(tok, "chat_template", None))


def _encode(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def _prefix_ids(tok) -> list[int]:
    """The special tokens the tokenizer puts in front of a sequence (BOS, if any)."""
    with_special, plain = list(tok.encode("x", add_special_tokens=True)), _encode(tok, "x")
    for i in range(len(with_special) - len(plain) + 1):
        if with_special[i:i + len(plain)] == plain:
            return with_special[:i]
    return []


def prompt_ids(tok, message: str) -> list[int]:
    if has_template(tok):
        text = tok.apply_chat_template([{"role": "user", "content": message}],
                                       tokenize=False, add_generation_prompt=True)
    else:
        text = CHATML.format(message=message)
    ids = _encode(tok, text)
    prefix = _prefix_ids(tok)
    return ids if ids[:len(prefix)] == prefix else prefix + ids


def target_ids(tok, output: str) -> list[int]:
    if tok.eos_token_id is None:
        raise ValueError("tokenizer has no eos_token_id; cannot terminate training targets")
    return _encode(tok, output) + [tok.eos_token_id]


def example_length(tok, instruction: str, output: str) -> int:
    return len(prompt_ids(tok, instruction)) + len(target_ids(tok, output))


def stop_ids(tok) -> list[int]:
    ids = [tok.eos_token_id]
    for marker in TURN_MARKERS:
        enc = _encode(tok, marker)
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def stop_strings(tok) -> list[str]:
    """Turn markers the tokenizer splits into several tokens, which only a
    string match can catch."""
    return [m for m in TURN_MARKERS[:2] if len(_encode(tok, m)) > 1]
