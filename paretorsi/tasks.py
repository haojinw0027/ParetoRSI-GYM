"""The three evaluation tasks: how an item becomes a prompt, and how a response
is judged against it.

    math   MATH-500            math_verify equivalence; the `ANSWER:` line as fallback
    mcq    GPQA, Medbullets,   the option letter, extracted by the most explicit
           SuperGPQA           marker present
    code   MBPP+, HumanEval+   the last fenced code block, run against the item's tests

`score` turns the per-sample verdicts into avg@k, pass@k and (mcq) maj@k.
"""

from __future__ import annotations

import os
import re
import resource
import subprocess
import sys
import tempfile
from collections import Counter

MATH_PROMPT = """\
Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command."""

MCQ_PROMPT = """\
{question}

{options}

Think step by step, then give your final answer.
The last line of your response must be of the form "ANSWER: $LETTER" (without quotes) where $LETTER is one of {letters}."""

CODE_PROMPT = """\
{prompt}

Your solution must satisfy this example check:
{assertion}

Answer with a single complete Python code block and nothing after it. The code
must be inside triple-backtick fences, BOTH the opening fence and the closing
fence, exactly in this shape:

```python
def name_from_the_task(...):
    ...
```

Nothing before the opening fence and nothing after the closing fence."""


def prompt(task: str, item: dict) -> str:
    if task == "math":
        return MATH_PROMPT.format(problem=item["problem"])
    if task == "mcq":
        letters = sorted(item["options"])
        return MCQ_PROMPT.format(question=item["question"],
                                 options="\n".join(f"{l}) {item['options'][l]}" for l in letters),
                                 letters="/".join(letters))
    if task == "code":
        return CODE_PROMPT.format(prompt=item["prompt"].strip(), assertion=item["assertions"][0])
    raise ValueError(task)


def judge(task: str, item: dict, text: str) -> tuple[bool, str | None, str]:
    """(correct, extracted answer or None, how it was extracted)."""
    if task == "math":
        return judge_math(text, item["answer"])
    if task == "mcq":
        letter, how = extract_letter(text, sorted(item["options"]))
        return letter == item["label"], letter, how
    if task == "code":
        code = extract_code(text)
        if code is None:
            return False, None, "none"
        return run_tests(code, item), None, "code"
    raise ValueError(task)


# ---------------------------------------------------------------------- math

_ANSWER_LINE = re.compile(r"ANSWER\s*\*{0,2}\s*[:：]\s*[*\s]*(.+)", re.I)
_ANSWER_PHRASE = re.compile(r"answer\s+is\s*[:：]?\s*[*\s]*(.+)", re.I)


def judge_math(text: str, gold: str) -> tuple[bool, str | None, str]:
    """math_verify over the whole response first; then the last `ANSWER:` (or
    "answer is") line, wrapped in $..$ because math_verify reads LaTeX only
    inside delimiters or \\boxed{}."""
    from math_verify import parse, verify
    try:
        g = parse(f"${gold}$")
        p = parse(text)
        if p and verify(g, p):
            return True, str(p[0]), "parsed"
        for pat in (_ANSWER_LINE, _ANSWER_PHRASE):
            lines = pat.findall(text)
            if not lines:
                continue
            cand = lines[-1].strip().rstrip(".").strip().strip("*").strip().strip("$")
            q = parse(f"${cand}$")
            if q:
                return bool(verify(g, q)), str(q[0]), "parsed"
        return False, (str(p[0]) if p else None), ("parsed" if p else "none")
    except (ValueError, TypeError, TimeoutError, RecursionError):
        return False, None, "none"


# ----------------------------------------------------------------------- mcq

def extract_letter(text: str, letters: list[str], tail_chars: int = 400) -> tuple[str | None, str]:
    """The most explicit marker wins: `ANSWER: X`, then \\boxed{X}, then
    "answer is X", each taken at its last occurrence.  Only without any of
    them does a bare option letter count, and only in the last `tail_chars`,
    because a letter inside prose ("vitamin D", "hepatitis B") is not an answer."""
    L = "".join(letters)
    patterns = (
        ("tagged", rf"ANSWER\s*\*{{0,2}}\s*[:：]\s*[*\s(]*([{L}])\b"),
        ("boxed", rf"\\boxed\{{\s*\(?([{L}])\)?\s*\}}"),
        ("phrase", rf"answer\s+is\s*:?\s*[*\s(]*([{L}])\b"),
    )
    for how, pat in patterns:
        hits = re.findall(pat, text, flags=re.I if how != "boxed" else 0)
        if hits:
            return hits[-1].upper(), how
    hits = re.findall(rf"(?<![A-Za-z])([{L}])(?![A-Za-z])", text[-tail_chars:])
    if hits:
        return hits[-1], "bare"
    return None, "none"


# ---------------------------------------------------------------------- code

_CODE_BLOCK = re.compile(r"```(?:python)?[ \t]*\n(.*?)```", re.S)
_FENCE_OPEN = re.compile(r"^[ \t]*```(?:python)?[ \t]*\n", re.M)
_FENCE_CLOSE = re.compile(r"\n?[ \t]*```[ \t]*$")


def extract_code(text: str) -> str | None:
    """The last complete fenced block.  Without one, an answer that defines a
    function is graded as written, minus a dangling opening or closing fence."""
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        return blocks[-1].strip()
    if "def " not in text:
        return None
    s = _FENCE_CLOSE.sub("", text.rstrip())
    opens = list(_FENCE_OPEN.finditer(s))
    if opens:
        s = s[opens[-1].end():]
    return s.strip() or None


def run_tests(code: str, item: dict, timeout_s: int = 20) -> bool:
    """Run the item's own test suite in a child process with CPU, memory and
    wall-clock limits.  This bounds runaway programs; it is not a security
    sandbox, so run evaluation where the model's code may do no harm."""
    script = "\n".join([*item.get("test_imports", []), code, item["test"]])

    def limits():
        for res, value in ((resource.RLIMIT_CPU, timeout_s), (resource.RLIMIT_AS, 4 << 30),
                           (resource.RLIMIT_CORE, 0)):
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):        # RLIMIT_AS is not enforceable on macOS
                pass

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution_test.py")
        with open(path, "w") as f:
            f.write(script)
        try:
            p = subprocess.run([sys.executable, path], cwd=tmp, capture_output=True,
                               timeout=timeout_s + 5, preexec_fn=limits,
                               env={"PATH": os.environ.get("PATH", ""), "HOME": tmp})
            return p.returncode == 0
        except subprocess.TimeoutExpired:
            return False


# ------------------------------------------------------------------- scoring

def score(task: str, items: list[dict], verdicts: dict[str, list[bool]],
          answers: dict[str, list[str | None]], k: int) -> dict:
    """avg@k: share of the k samples correct, averaged over items.
    pass@k: share of items with at least one correct sample.
    maj@k (mcq): the extracted letters vote; the item counts only if the gold
    letter is the unique mode.  Abstentions do not vote, and a tie loses."""
    n = len(items)
    out = {f"avg@{k}": sum(sum(verdicts[i["id"]]) / k for i in items) / n,
           f"pass@{k}": sum(any(verdicts[i["id"]]) for i in items) / n}
    if task == "mcq":
        def majority_correct(item):
            top = Counter(a for a in answers[item["id"]] if a).most_common(2)
            return bool(top) and top[0][0] == item["label"] and (len(top) == 1 or top[0][1] > top[1][1])
        out[f"maj@{k}"] = sum(majority_correct(i) for i in items) / n
    return {name: round(v, 4) for name, v in out.items()}
