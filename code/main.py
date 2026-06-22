#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent
Entry point: python main.py
"""

import os
import sys
import argparse
from pathlib import Path
from agent import SupportTriageAgent
from logger import Logger


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


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent"
    )
    parser.add_argument(
        "--input",
        default="../support_tickets/support_tickets.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        default="../support_tickets/output.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--corpus",
        default="../data",
        help="Path to support corpus directory",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample_support_tickets.csv instead",
    )
    args = parser.parse_args()

    load_env_file(Path(__file__).with_name('.env'))
    logger = Logger()
    logger.log_session_start()

    if args.sample:
        input_path = args.input.replace("support_tickets.csv", "sample_support_tickets.csv")
    else:
        input_path = args.input

    print("\n" + "=" * 60)
    print("  Multi-Domain Support Triage Agent")
    print("=" * 60)
    print(f"  Input:  {input_path}")
    print(f"  Output: {args.output}")
    print(f"  Corpus: {args.corpus}")
    print("=" * 60 + "\n")

    agent = SupportTriageAgent(corpus_dir=args.corpus)
    agent.process_csv(input_path=input_path, output_path=args.output)

    print("\n" + "=" * 60)
    print("  Done! Results written to:", args.output)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()