# Multi-Domain Support Triage Agent

A terminal-based support triage agent that processes support tickets for **HackerRank**, **Claude**, and **Visa** using retrieval-augmented reasoning over the provided corpus.

---

## Architecture

```
main.py          → CLI entry point and argument parsing
agent.py         → Agent orchestration: retrieval → prompt → LLM call → CSV output
retriever.py     → FAISS vector retrieval over the local corpus
router.py        → Company inference plus high-risk signal detection
logger.py        → Append-only run/logging helper for AGENTS.md compliance
```

### Design Decisions

| Component | Choice | Reason |
|---|---|---|
| Retrieval | FAISS + local embeddings | Ground answers in the corpus; reproducible; no external vector DB needed |
| LLM | Gemini via `google-genai` | Primary reasoning engine with structured JSON output |
| Output format | Strict JSON parsing | Avoids markdown or free-form variations |
| Escalation | Keyword pre-screen + model decision | Layered handling for high-risk or sensitive tickets |
| Deployment | Env vars only | No hardcoded keys; API credentials come from `GEMINI_API_KEY` / `GEMINI_API_KEYS`

---

## Setup

### 1. Install dependencies

From the `code` folder, run:

```bash
./run.sh
```

This installs the pinned versions into the `code/rag_env` environment and then runs `main.py`.

If you prefer to install manually from inside the environment:

```bash
cd code
./rag_env/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Configure credentials

Copy `code/.env.example` to `code/.env` or set the environment variables directly.

Required:
- `GEMINI_API_KEY` or `GEMINI_API_KEYS`

Optional:
- `ANTHROPIC_API_KEY`
- `GLM_API_KEY`
- `LOCAL_AI_URL`

---

## Running

### Default full ticket run

```bash
python main.py
```

### Run the sample ticket set

```bash
python main.py --sample
```

### Custom paths

```bash
python main.py \
  --input ../support_tickets/support_tickets.csv \
  --output ../support_tickets/output.csv \
  --corpus ../data
```

---

## Output Schema

| Column | Values |
|---|---|
| `issue` | Original ticket text |
| `subject` | Original subject |
| `company` | Original company field |
| `response` | User-facing answer grounded in corpus |
| `product_area` | Support category |
| `status` | `replied` or `escalated` |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |
| `justification` | Agent reasoning or escalation rationale |

---

## Escalation Logic

A ticket is escalated when it signals:
- fraud, unauthorized transactions, or account compromise
- billing disputes, refunds, or legal concerns
- score manipulation, unfair grading, or restore-access requests
- platform-wide outages or missing corpus coverage for sensitive issues

---

## Chat Transcript Logging

Log file location: `%USERPROFILE%\hackerrank_orchestrate\log.txt`

The repository is set up to append session and turn entries for AI collaboration compliance.

---

## Reproducibility

- Primary model: Gemini 2.5 flash via `google-genai`
- Retrieval: deterministic FAISS corpus search
- Secrets: read only from environment variables
- Dependencies pinned in `requirements.txt`
