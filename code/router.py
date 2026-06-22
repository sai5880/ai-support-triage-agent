"""
router.py — Company inference and pre-flight risk signals.

The Router does a lightweight keyword pass BEFORE calling the LLM:
  - Infers which company the ticket belongs to when company == None / empty
  - Flags high-risk signals (fraud, legal, account compromise, etc.)
    so the agent prompt can be informed

The actual status/request_type decisions are made by the LLM, but
the router's signal is included in the prompt context.
"""

import re
from typing import Optional


# Keywords for company inference
COMPANY_SIGNALS = {
    "HackerRank": [
        "hackerrank", "hacker rank", "test", "assessment", "coding challenge",
        "recruiter", "candidate", "score", "submission", "plagiarism", "proctoring",
        "invite", "interview", "coding test", "hackerrank test", "test link",
        "test expir", "test active", "question bank", "code pair",
    ],
    "Claude": [
        "claude", "anthropic", "claude.ai", "claude pro", "claude team",
        "workspace", "conversation", "artifact", "context window", "mcp",
        "claude api", "subscription", "plan upgrade", "claude account",
        "api key", "usage limit", "message limit",
    ],
    "Visa": [
        "visa", "visa card", "credit card", "debit card", "transaction",
        "payment", "merchant", "chargeback", "refund", "fraud", "unauthorized",
        "visa account", "card number", "atm", "pin", "contactless",
        "visa checkout", "dispute", "bank", "statement",
    ],
}

# High-risk patterns that should bias toward escalation
HIGH_RISK_PATTERNS = [
    r"\bfraud\b",
    r"\bunauthorized\b",
    r"\bchargeback\b",
    r"\bstolen\b",
    r"\bhacked\b",
    r"\bcompromised\b",
    r"\blegal\b",
    r"\blawsuit\b",
    r"\bban\b.*\bseller\b",
    r"\bban\b.*\bmerchant\b",
    r"\bincrease.*score\b",
    r"\bchange.*score\b",
    r"\bmanipulate\b",
    r"\bscore.*unfair\b",
    r"\brestore.*access\b",
    r"\bsite.*down\b",
    r"\bpages.*not.*accessible\b",
    r"\bnot.*admin\b",
    r"\bwithout.*permission\b",
]


class Router:
    def infer_company(self, issue: str, subject: str, company: str) -> str:
        """Return the most likely company, or original value if already set."""
        if company and company.strip().lower() not in ("none", "", "unknown"):
            # Normalize casing
            for name in COMPANY_SIGNALS:
                if name.lower() == company.strip().lower():
                    return name
            return company.strip()

        text = f"{issue} {subject}".lower()
        scores = {name: 0 for name in COMPANY_SIGNALS}
        for name, keywords in COMPANY_SIGNALS.items():
            for kw in keywords:
                if kw in text:
                    scores[name] += 1

        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return "Unknown"
        return best

    def is_high_risk(self, issue: str, subject: str) -> bool:
        """Return True if ticket contains high-risk signals."""
        text = f"{issue} {subject}".lower()
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def get_risk_note(self, issue: str, subject: str) -> Optional[str]:
        """Return a human-readable risk note, or None."""
        text = f"{issue} {subject}".lower()
        matched = []
        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, text):
                matched.append(pattern.replace(r"\b", "").replace(".*", " "))
        if matched:
            return f"High-risk signals detected: {', '.join(matched[:3])}"
        return None