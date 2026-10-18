"""Launching the curation agent, confining it, and auditing what it did.

Three agents are supported: claude-code (`claude --print`), codex (`codex exec`)
and cheetahclaws (a runner script you supply with `session.py init
--cheetah-runner`; it is called as `python runner --workspace W --session S
--model M` with the prompt on stdin, and prints one JSON object per tool call,
{"type": "tool_start", "name": ..., "input": {...}}).  Each keeps one conversation for the
whole run: its state lives in <run>/agent_home/, so a later round or a relaunch
resumes it.

The agent runs under bubblewrap.  Its filesystem is the system directories
read-only, its own toolchain read-only, the workspace and agent_home
read-write, and the `pb` client; $HOME and /tmp are empty.  The repository
(and with it the benchmark files), the rest of the run directory and every
other run do not exist inside the sandbox.  The network is shared: the agent
needs the gateway and its model API, and it may search the web for sources.
Data reaches training only through the gateway, so a file the agent fetches
by other means cannot be trained on; such commands are logged as `bash_net`.
Its environment is built from scratch: no Hugging Face token, only the
credential its own model API needs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .common import ROOT, round_dir
from .store import first_hit

AGENTS = ("claude-code", "codex", "cheetahclaws")
SECRETS = {"claude-code": ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"),
           "codex": ("OPENAI_API_KEY",),
           "cheetahclaws": ("DEEPSEEK_API_KEY",)}
SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt")
NET_TOOLS = ("curl ", "wget ", "requests.", "urllib", "httpx", "load_dataset", "hf_hub_download",
             "huggingface-cli", "git clone", "pip install", "pip download")


# ------------------------------------------------------------------ set-up

def prepare(run: Path, agent: str, *, codex_effort: str, cheetah_runner: str | None) -> None:
    """Create the agent's persistent home inside the run and copy in its login."""
    home = Path(run) / "agent_home"
    home.mkdir(exist_ok=True)
    if agent == "claude-code":
        d = home / "claude"
        d.mkdir(exist_ok=True)
        for src in (Path.home() / ".claude" / ".credentials.json", Path.home() / ".claude.json"):
            if src.exists():
                shutil.copy2(src, d / src.name)
        (home / "claude_session").write_text(str(uuid.uuid4()))
    elif agent == "codex":
        d = home / "codex"
        d.mkdir(exist_ok=True)
        src = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        for name in ("auth.json", "installation_id"):
            if (src / name).exists():
                shutil.copy2(src / name, d / name)
        (d / "config.toml").write_text(f'model_reasoning_effort = "{codex_effort}"\nweb_search = "live"\n')
    elif agent == "cheetahclaws":
        if not cheetah_runner or not Path(cheetah_runner).is_file():
            raise SystemExit("cheetahclaws needs --cheetah-runner <script>")
        (home / "cheetah").mkdir(exist_ok=True)
        (Path(run) / "bin").mkdir(exist_ok=True)
        shutil.copy2(cheetah_runner, Path(run) / "bin" / "cheetah_runner.py")
    else:
        raise ValueError(f"agent must be one of {AGENTS}")


def install_pb(run: Path) -> None:
    b = Path(run) / "bin"
    b.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name("pb.py"), b / "pb.py")
    (b / "pb").write_text('#!/bin/sh\nexec python3 "$(dirname "$0")/pb.py" "$@"\n')
    (b / "pb").chmod(0o755)


# ----------------------------------------------------------------- command

def command(run: Path, agent: str, model: str | None, prompt: str, resume: bool, usd: float,
            cheetah_python: str) -> tuple[list[str], dict, str | None]:
    """(argv, extra environment, stdin) for one launch."""
    home = Path(run) / "agent_home"
    ws = Path(run) / "workspace"
    if agent == "claude-code":
        sid = (home / "claude_session").read_text().strip()
        argv = ["claude", "--print", prompt, *(["--resume", sid] if resume else ["--session-id", sid]),
                "--permission-mode", "bypassPermissions",
                # an allowlist: no WebFetch, so no page text reaches the agent except through search
                "--tools", "Bash,Read,Edit,Write,Glob,Grep,WebSearch",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--output-format", "stream-json", "--verbose", "--max-budget-usd", f"{usd:.2f}"]
        return argv + (["--model", model] if model else []), {"CLAUDE_CONFIG_DIR": str(home / "claude")}, None
    if agent == "codex":
        # bubblewrap is the sandbox, so codex's own is bypassed
        argv = ["codex", "exec", "--json", "--skip-git-repo-check", "-C", str(ws),
                "--dangerously-bypass-approvals-and-sandbox", *(["-m", model] if model else [])]
        return argv + (["resume", "--last", prompt] if resume else [prompt]), {"CODEX_HOME": str(home / "codex")}, None
    argv = [cheetah_python, str(Path(run) / "bin" / "cheetah_runner.py"), "--workspace", str(ws),
            "--session", str(home / "cheetah" / "session.json"),
            "--model", model or "deepseek/deepseek-v4-flash"]
    return argv, {}, prompt                          # the prompt goes on stdin, never into argv


def _toolchain(paths: list[str]) -> list[str]:
    """Install prefixes of the executables the agent needs, to bind read-only."""
    out = set()
    for p in paths:
        exe = shutil.which(p) or p
        real = Path(os.path.realpath(exe))
        for cand in (Path(exe), real):
            if "node_modules" in cand.parts:
                cand = Path(*cand.parts[:cand.parts.index("node_modules") - 1])
            else:
                cand = cand.parent.parent
            if not str(cand).startswith(SYSTEM_DIRS):
                out.add(str(cand))
    return sorted(out)


def sandbox(run: Path, argv: list[str], env: dict, extra_ro: list[str]) -> list[str]:
    if shutil.which("bwrap") is None:
        raise SystemExit("bubblewrap (bwrap) is required to confine the agent; "
                         "pass --no-sandbox to run unconfined (recorded in meta.json)")
    run = Path(run).resolve()
    ro = _toolchain([argv[0], "node", "python3"]) + extra_ro
    for hidden in (ROOT.resolve(), run):
        for b in (*SYSTEM_DIRS, *ro):
            if str(hidden) == b or str(hidden).startswith(b.rstrip("/") + "/"):
                raise SystemExit(f"{hidden} lies under {b}, which the sandbox exposes; move the repository "
                                 f"or the run directory elsewhere")
    args = ["bwrap", "--die-with-parent", "--unshare-ipc", "--unshare-uts", "--clearenv"]
    for d in SYSTEM_DIRS:
        if os.path.exists(d):
            args += ["--ro-bind", d, d]
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", env["HOME"]]
    for d in ro:
        args += ["--ro-bind", d, d]
    args += ["--bind", str(run / "workspace"), str(run / "workspace"),
             "--bind", str(run / "agent_home"), str(run / "agent_home"),
             "--ro-bind", str(run / "bin"), str(run / "bin"), "--chdir", str(run / "workspace")]
    for k, v in env.items():
        args += ["--setenv", k, v]
    return args + ["--"] + argv


def launch(run: Path, round_no: int, agent: str, *, model: str | None, prompt: str, resume: bool,
           usd: float, deadline: float, gateway_url: str, attempt: int, confine: bool,
           cheetah_python: str, extra_ro: list[str]) -> int:
    argv, extra, stdin = command(run, agent, model, prompt, resume, usd, cheetah_python)
    env = {"HOME": os.environ.get("HOME", "/home/agent"), "LANG": os.environ.get("LANG", "C.UTF-8"),
           "TERM": "dumb", "PB_GATEWAY_URL": gateway_url,
           "PATH": os.pathsep.join([str(Path(run).resolve() / "bin"), os.environ.get("PATH", "/usr/bin:/bin")]),
           **extra, **{k: os.environ[k] for k in SECRETS[agent] if k in os.environ}}
    full = sandbox(run, argv, env, extra_ro) if confine else argv
    log = round_dir(run, round_no) / f"agent_{attempt}.jsonl"
    with log.open("w") as f:
        p = subprocess.Popen(full, cwd=Path(run) / "workspace", env=None if confine else env,
                             stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.communicate(stdin.encode() if stdin else None, timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 15)
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 9)
                p.wait()
    return p.returncode


# ------------------------------------------------------------------- audit

def _tool_calls(log: Path):
    """(tool, input) from a Claude Code stream-json, Codex --json or cheetah_runner log."""
    for line in log.read_text(errors="replace").split("\n"):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "tool_start":                                  # cheetah_runner
            yield ev.get("name"), ev.get("input") if isinstance(ev.get("input"), dict) else {}
        elif ev.get("type") == "item.completed":                            # codex
            it = ev.get("item") or {}
            if it.get("type") == "command_execution":
                yield "Bash", {"command": it.get("command")}
            elif "search" in str(it.get("type")):
                yield "WebSearch", {"query": it.get("query") or it.get("text")}
        else:                                                               # claude code
            for block in ((ev.get("message") or {}).get("content") or []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    yield block.get("name"), block.get("input") or {}


def audit_transcript(run: Path, round_no: int, grams: set[str], n: int) -> dict:
    """Web searches and network commands, and any of them carrying evaluation text."""
    searches, net = [], []
    for log in sorted(round_dir(run, round_no).glob("agent_*.jsonl")):
        for tool, inp in _tool_calls(log):
            if tool in ("WebSearch", "web_search") and inp.get("query"):
                searches.append(str(inp["query"]))
            elif tool in ("Bash", "bash") and inp.get("command"):
                cmd = str(inp["command"])
                if any(t in cmd for t in NET_TOOLS) and not cmd.lstrip().startswith("pb "):
                    net.append(cmd[:300])
    hits = [{"text": s[:200], "ngram": g} for s in searches + net if (g := first_hit(s, grams, n))]
    return {"searches": searches, "bash_net": net, "lookup_violations": hits}
