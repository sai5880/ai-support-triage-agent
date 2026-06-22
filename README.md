# Multi-Domain AI Support Triage Agent

An intelligent, terminal-based support ticket triage system that automatically resolves customer issues across **three product ecosystems** — HackerRank, Claude, and Visa — using retrieval-augmented generation (RAG), vector search, and multi-LLM orchestration.

**What it solves:** Support teams spend hours manually triaging tickets across knowledge bases. This agent reads each ticket, retrieves the most relevant documentation from a local corpus, classifies the issue (product issue, bug, feature request, or invalid), determines whether to reply or escalate, and generates a grounded, safe response — all without hallucinating policies or guessing.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Google Gemini 2.5 Flash (primary), GLM via Nvidia API (fallback), Claude (fallback) | Multi-provider resilience with automatic key rotation |
| **Vector Search** | FAISS (Facebook AI Similarity Search) | Fast, deterministic retrieval over ~3000 chunks |
| **Embeddings** | BAAI/bge-small-en-v1.5 via OpenVINO on Intel Iris Xe GPU | Local, hardware-accelerated inference |
| **Orchestration** | Python — custom pipeline (no framework) | Full control over retrieval, routing, and output |
| **Structured Output** | JSON schema with strict parsing | Enforces exact output contract every time |
| **Risk Detection** | Rule-based pre-screen + LLM judgment | Layered safety for fraud, legal, and sensitive cases |
| **Company Router** | Keyword-weighted inference | Determines domain when company field is missing |

---

## Architecture

```
main.py          → CLI entry point, argument parsing, env loading
agent.py         → Agent pipeline: retrieve → build prompt → call LLM → parse JSON → write CSV
retriever.py     → FAISS index builder + chunked retrieval over local corpus
router.py        → Company inference + high-risk signal detection
logger.py        → Session/turn logging for collaboration compliance
```

### Pipeline

1. **Load** ticket CSV into rows
2. **Route** — infer company (HackerRank/Claude/Visa) from content if not provided, detect high-risk signals
3. **Retrieve** — embed query with OpenVINO on GPU, search FAISS index for top-3 relevant corpus passages
4. **Prompt** — build a prompt with ticket + corpus context, instructing strict JSON output
5. **Generate** — call Gemini 2.5 Flash with `response_mime_type=application/json` and system prompt enforcing the output schema
6. **Fallback chain** — if Gemini fails, rotate through API keys, then fall back to GLM → Claude → Local AI → default escalation
7. **Parse** — extract and validate JSON, ensure all 5 required keys exist with valid values
8. **Write** — append result row to output CSV

---

## Key Features

- **Zero-hallucination guarantee** — responses are grounded in the provided corpus, never generated from model knowledge
- **Automatic key rotation** — supports multiple Gemini API keys for rate-limit resilience
- **Deterministic retrieval** — FAISS with seeded sampling for reproducible results
- **Multi-provider fallback** — Gemini → GLM → Claude → Local AI → safe escalation
- **Risk-aware escalation** — fraud, legal, stolen cards, score manipulation, platform outages all trigger escalation
- **Company inference** — automatically detects the right domain even when the company field is empty
- **Intel GPU acceleration** — embeddings run on Iris Xe via OpenVINO for fast local inference
- **No external API calls for knowledge** — everything uses the local support corpus

---

## Setup

### Prerequisites
- Python 3.10+
- Intel Iris Xe GPU or compatible (for OpenVINO embedding — optional, falls back gracefully)

### Install & Run

```bash
cd code
./run.sh
```

Or manually:
```bash
cd code
pip install -r requirements.txt
python main.py
```

### Configuration

Copy `.env.example` to `.env` and set your API keys:

```bash
cp .env.example .env
```

Required: `GEMINI_API_KEY` or `GEMINI_API_KEYS` (comma-separated for rotation).

Optional: `ANTHROPIC_API_KEY`, `GLM_API_KEY`, `LOCAL_AI_URL`.

---

## Output Schema

| Column | Values |
|---|---|
| `status` | `replied` — agent answered from corpus / `escalated` — forwarded to human |
| `product_area` | Most relevant support category |
| `response` | User-facing answer grounded in documentation |
| `justification` | Why the agent made that decision |
| `request_type` | `product_issue`, `feature_request`, `bug`, or `invalid` |

---

## Sample Results

| Ticket | Status | Product Area |
|---|---|---|
| "How long do tests stay active?" | replied | screen |
| "Site is down, pages not accessible" | escalated | General Support |
| "Lost Claude access after IT removed my seat" | replied | admin-management |
| "My Visa card was stolen, what should I do?" | escalated | General Support |

---

## Built For

This project was submitted for the **HackerRank Orchestrate** 24-hour hackathon (May 1–2, 2026). The challenge was to build a terminal-based AI agent that triages real support tickets using only a provided support corpus, with strict output contracts and escalation rules.

[Problem Statement](./problem_statement.md) · [Evaluation Criteria](./evalutation_criteria.md)

---

## Project Structure

```
.
├── code/                    # Agent source code
│   ├── main.py             # Entry point
│   ├── agent.py            # Pipeline orchestration
│   ├── retriever.py        # FAISS vector retrieval
│   ├── router.py           # Company + risk detection
│   ├── logger.py           # Session logging
│   ├── test_agent.py       # Smoke test suite
│   ├── scrape_corpus.py    # Corpus builder
│   └── requirements.txt    # Pinned dependencies
├── data/                   # Support corpus (3 domains)
├── support_tickets/        # Input CSVs + agent output
└── .env.example            # Environment variable template
```

---

## License

MIT
