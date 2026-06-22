"""
logger.py — Append-only log writer per AGENTS.md §2 and §5.

Log file: $HOME/hackerrank_orchestrate/log.txt
"""

import os
import pathlib
import subprocess
from datetime import datetime, timedelta, timezone


LOG_DIR = pathlib.Path.home() / "hackerrank_orchestrate"
LOG_FILE = LOG_DIR / "log.txt"

AGENT_NAME = "support-triage-agent"
REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
DEADLINE_IST = timezone(timedelta(hours=5, minutes=30))
DEADLINE_ISO = "2026-05-02T11:00:00+05:30"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _time_remaining(include_deadline: bool = False) -> str:
    now = datetime.now(timezone.utc).astimezone(DEADLINE_IST)
    deadline = datetime(2026, 5, 2, 11, 0, 0, tzinfo=DEADLINE_IST)
    delta = deadline - now
    if delta.total_seconds() < 0:
        remaining = "0d 0h 0m"
    else:
        days = delta.days
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        remaining = f"{days}d {hours}h {minutes}m"
    if include_deadline:
        return f"{remaining} until {DEADLINE_ISO}"
    return remaining


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=pathlib.Path(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        return branch if branch else "unknown"
    except Exception:
        return "unknown"


def _ensure_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.write_text("", encoding="utf-8")


class Logger:
    def log_session_start(self):
        _ensure_log()
        branch = _git_branch()
        entry = f"""
## [{_now()}] SESSION START
Agent: {AGENT_NAME}
Repo Root: {REPO_ROOT}
Branch: {branch}
Worktree: main
Parent Agent: none
Language: py
Time Remaining: {_time_remaining()}
"""
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)

    def log_turn(self, title: str, user_prompt: str, summary: str, actions: list):
        _ensure_log()
        branch = _git_branch()
        # Redact anything that looks like an API key
        safe_prompt = _redact_secrets(user_prompt)
        actions_str = "\n".join(f"* {a}" for a in actions) if actions else "* none"
        entry = f"""
## [{_now()}] {title[:80]}
User Prompt (verbatim, secrets redacted):
{safe_prompt[:500]}

Agent Response Summary:
{summary}

Actions:
{actions_str}

Context:
tool={AGENT_NAME}
branch={branch}
repo_root={REPO_ROOT}
worktree=main
parent_agent=none
"""
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)

    def log_agreement(self, language: str = "py"):
        _ensure_log()
        entry = f"""
## [{_now()}] ONBOARDING COMPLETE
AGREEMENT RECORDED: {REPO_ROOT}
Agent: {AGENT_NAME}
Language: {language}
System Time: {_now()}
Time Remaining: {_time_remaining(include_deadline=True)}
"""
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry)


def _redact_secrets(text: str) -> str:
    import re
    # Redact anything resembling an API key (long alphanumeric strings)
    text = re.sub(r"sk-[A-Za-z0-9\-_]{20,}", "[REDACTED]", text)
    text = re.sub(r"key[=:\s]+[A-Za-z0-9\-_]{20,}", "key=[REDACTED]", text, flags=re.IGNORECASE)
    return text