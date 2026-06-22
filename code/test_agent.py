"""
test_agent.py — Quick smoke test using the three provided sample tickets.

Run: python test_agent.py

Uses the configured corpus to verify the ticket handling pipeline.
For real production runs, use: python main.py
"""

import os
import sys
from pathlib import Path


def load_env_file(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_env_file(Path(__file__).with_name('.env'))

# Ensure Gemini credentials are available
if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GEMINI_API_KEYS"):
    print("ERROR: Set GEMINI_API_KEY or GEMINI_API_KEYS before running.")
    sys.exit(1)

from agent import SupportTriageAgent

SAMPLE_TICKETS = [
    {
        "issue": "I notice that people I assigned the test in October of 2025 have not received new tests. How long do the tests stay active in the system.",
        "subject": "Test Active in the system",
        "company": "HackerRank",
        "expected_status": "replied",
        "expected_area": "screen",
    },
    {
        "issue": "site is down & none of the pages are accessible",
        "subject": "",
        "company": "None",
        "expected_status": "escalated",
        "expected_area": "",
    },
    {
        "issue": "I lost access to my Claude team workspace after our IT admin removed my seat. Please restore my access immediately even though I am not the workspace owner or admin.",
        "subject": "Claude access lost",
        "company": "Claude",
        "expected_status": "escalated",
        "expected_area": "",
    },
]


def main():
    print("\n=== Smoke Test: 3 Sample Tickets ===\n")
    agent = SupportTriageAgent(corpus_dir="../data")

    for i, ticket in enumerate(SAMPLE_TICKETS, 1):
        print(f"--- Ticket {i} ---")
        print(f"Issue: {ticket['issue'][:80]}...")
        result = agent._process_ticket(
            issue=ticket["issue"],
            subject=ticket["subject"],
            company=ticket["company"],
        )
        print(f"  status:       {result['status']}  (expected: {ticket['expected_status']})")
        print(f"  product_area: {result['product_area']}")
        print(f"  request_type: {result['request_type']}")
        print(f"  response:     {result['response'][:120]}...")
        print(f"  justification:{result['justification'][:100]}...")
        ok = result['status'] == ticket['expected_status']
        print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
        print()


if __name__ == "__main__":
    main()