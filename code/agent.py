"""
agent.py — Core support triage agent.

Architecture:
  1. Corpus Loader  — handled inside Retriever
  2. Retriever      — FAISS vector retrieval over corpus embeddings
  3. Router         — classifies company, product_area, request_type, risk
  4. Responder      — Gemini | GLM | Claude | Local AI
  5. Output Writer  — writes structured CSV rows
"""

import os
import csv
import re
import json
import time
import requests

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from google import genai
from google.genai import types as genai_types

from retriever import Retriever
from router import Router


SYSTEM_PROMPT = """You are a support triage agent. Output ONLY a JSON object. Nothing else.

When answering a user's issue based on the provided context, you MUST adhere to the following rules:

1. READ ALL CONTEXT: Do not just use the first document. Read all provided chunks to find the one that best matches the user's specific symptoms. 
2. MATCH THE SYMPTOM: If the user says they cannot send/assign new tests, find the document that explicitly mentions the "Invite" button being disabled or not being able to invite new candidates.
3. PROVIDE STEP-BY-STEP INSTRUCTIONS: If the correct document contains a numbered list or UI navigation path (e.g., "Settings > General"), you MUST include those exact steps in your response. Do not just tell the user "you can modify it" — tell them EXACTLY how to do it.
4. ANSWER DIRECTLY: Always explicitly answer the user's primary question (e.g., "How long do tests stay active?").

ABSOLUTE OUTPUT RULE:
- Return EXACTLY this JSON structure with EXACTLY these 5 keys
- No extra keys. No markdown. No explanation. No text before or after.

{
  "status": "replied" or "escalated",
  "product_area": "<category>",
  "response": "<user-facing answer>",
  "justification": "<1-2 sentences>",
  "request_type": "product_issue" or "feature_request" or "bug" or "invalid"
}

ANY response not matching this exact structure is WRONG.

status: whether the agent should answer directly or escalate
product_area: the most relevant support category or domain area
response: a user-facing answer grounded in the support corpus
justification: a concise explanation of the decision & response
request_type: the best-fit request classification

avoid unsupported claims or hallucinated policies
escalate high-risk, sensitive, or unsupported cases when appropriate

====================
WHEN TO REPLY
====================
- FAQ, how-to, configuration, account management
- Remove/add users, delete account, settings
- Product usage questions, general queries
- Out of scope → reply politely saying it is outside supported scope

====================
WHEN TO ESCALATE
====================
- Fraud, stolen card, unauthorized transactions
- Billing disputes, refunds
- Legal or enforcement requests
- Score manipulation complaints
- Platform-wide outage or site down


====================
SOURCE RULE
====================
Use ONLY the provided corpus passages. If corpus has no answer:
- Low risk → status: replied
- High risk, sensitive, or out-of-scope → status: escalated

====================
REMINDER
====================
Your entire output must be a single JSON object.
EXACTLY 5 keys: status, product_area, response, justification, request_type.
No other keys. No markdown. No extra text.
"""

VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}


def _load_gemini_keys():
    keys = [
        k.strip()
        for k in os.environ.get("GEMINI_API_KEYS", "").split(",")
        if k.strip()
    ]
    if not keys:
        primary_key = os.environ.get("GEMINI_API_KEY")
        if primary_key:
            keys = [primary_key.strip()]
    return keys


class SupportTriageAgent:
    def __init__(self, corpus_dir: str):
            self.corpus_dir = corpus_dir
            self.retriever = Retriever(corpus_dir)
            self.router = Router()
            self.mode = "gemini"

            self.api_keys = _load_gemini_keys()
            if not self.api_keys:
                raise EnvironmentError(
                    "GEMINI_API_KEY or GEMINI_API_KEYS must be set before running."
                )
            self.current_key_idx = 0

            self._init_gemini_client()
            print(f"[INFO] Using Gemini (gemini-2.5-flash) with {len(self.api_keys)} key(s)")

    def _init_gemini_client(self):
        """Initializes the Gemini client with the currently active key."""
        active_key = self.api_keys[self.current_key_idx]
        self.gemini_client = genai.Client(api_key=active_key)

    def _rotate_key(self):
        """Moves to the next API key in the list and re-initializes the client."""
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        print(f"[INFO] Rotating to API Key #{self.current_key_idx + 1}...")
        self._init_gemini_client()

    # -------------------------
    # Main CSV processing
    # -------------------------
    def process_csv(self, input_path: str, output_path: str):
        rows = self._read_csv(input_path)
        print(f"Loaded {len(rows)} tickets\n")

        results = []
        for i, row in enumerate(rows, 1):
            issue = row.get("Issue") or row.get("issue") or ""
            subject = row.get("Subject") or row.get("subject") or ""
            company = row.get("Company") or row.get("company") or ""

            print(f"[{i}/{len(rows)}] {issue[:80]}...")

            result = self._process_ticket(issue, subject, company)
            result["issue"] = issue
            result["subject"] = subject
            result["company"] = company
            print(result)

            results.append(result)
            if i < len(rows):
                print("[INFO] Waiting 20 seconds before next ticket...")
                time.sleep(20)

        self._write_csv(output_path, results)

    # -------------------------
    # Single ticket pipeline
    # -------------------------
    def _process_ticket(self, issue: str, subject: str, company: str) -> dict:
        inferred_company = self.router.infer_company(issue, subject, company)
        passages = self.retriever.retrieve(issue, subject, inferred_company, top_k=3)
        corpus_context = self._format_passages(passages)
        prompt = self._build_prompt(issue, subject, inferred_company, corpus_context)
        return self._call_gemini(prompt)

    # -------------------------
    # Prompt builder
    # -------------------------
    def _build_prompt(self, issue, subject, company, context):
        return f"""SUPPORT TICKET:
Company: {company or 'Unknown'}
Subject: {subject or '(none)'}
Issue: {issue}

RELEVANT SUPPORT CORPUS:
{context if context else "(No relevant passages found)"}

Return ONLY JSON as specified."""

    # -------------------------
    # Gemini
    # -------------------------

    def _call_gemini(self, prompt: str) -> dict:
            fallback = self._fallback()
            max_retries = 3 # Will try all 3 keys before giving up

            for attempt in range(max_retries):
                try:
                    response = self.gemini_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=genai_types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.0,
                            top_k=1,
                            top_p=0.1,
                            max_output_tokens=8120,
                            response_mime_type="application/json",
                        ),
                    )
                    raw = response.text.strip()
                    print(f"[GEMINI RAW] {raw}")
                    return self._safe_parse(raw, fallback)

                except Exception as e:
                    print(f"[WARN] Gemini call failed (Attempt {attempt + 1}/{max_retries}): {e}")

                    # If we haven't run out of retries yet, wait and swap keys
                    if attempt < max_retries - 1:
                        print("[INFO] Waiting 10 seconds before retrying with next key...")
                        time.sleep(10)
                        self._rotate_key()
                    else:
                        print("[ERROR] All retries exhausted. Using fallback response.")
                        return fallback
    # -------------------------
    # GLM via Nvidia API (backup)
    # -------------------------
    def _call_glm(self, prompt: str) -> dict:
        fallback = self._fallback()
        api_key = os.environ.get("GLM_API_KEY")
        if not api_key:
            print("[WARN] GLM_API_KEY not set; skipping GLM fallback.")
            return fallback

        try:
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model="z-ai/glm-5.1",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                top_p=1,
                max_tokens=512,
                stream=False,
            )
            raw = completion.choices[0].message.content.strip()
            print(f"[GLM RAW] {raw}")
            return self._safe_parse(raw, fallback)

        except Exception as e:
            print(f"[ERROR] GLM failed: {e}")
            return fallback

    # -------------------------
    # Claude API (backup)
    # -------------------------
    def _call_claude(self, prompt: str) -> dict:
        fallback = self._fallback()
        try:
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            print(f"[CLAUDE RAW] {raw}")
            return self._safe_parse(raw, fallback)

        except Exception as e:
            print(f"[ERROR] Claude failed: {e}")
            return fallback

    # -------------------------
    # Local AI (backup)
    # -------------------------
    def _call_local_ai(self, prompt: str) -> dict:
        fallback = self._fallback()
        try:
            url = os.environ.get("LOCAL_AI_URL", "http://192.168.1.17:8000/chat")
            res = requests.post(
                url,
                json={"system": SYSTEM_PROMPT, "prompt": prompt},
                timeout=6000,
            )
            print(f"[LOCAL RAW] {res.text}")
            if res.status_code != 200:
                return fallback
            raw = res.json().get("response", "").strip()
            return self._safe_parse(raw, fallback)

        except Exception as e:
            print(f"[ERROR] Local AI failed: {e}")
            return fallback

    # -------------------------
    # JSON parser — strict
    # -------------------------
    def _safe_parse(self, raw: str, fallback: dict) -> dict:
        raw = re.sub(r"^```json\s*", "", raw.strip())
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)

        try:
            parsed = json.loads(raw)
        except Exception as e:
            print(f"[WARN] JSON parse failed: {e} | raw: {raw[:200]}")
            return fallback

        status_raw = str(parsed.get("status", "")).lower()
        if "escalat" in status_raw:
            status = "escalated"
        elif "repl" in status_raw:
            status = "replied"
        else:
            status = fallback["status"]

        rt = str(parsed.get("request_type", "")).lower()
        request_type = rt if rt in VALID_REQUEST_TYPE else fallback["request_type"]

        response_text = (
            parsed.get("response") or
            parsed.get("suggested_action") or
            parsed.get("answer") or
            parsed.get("message") or
            fallback["response"]
        )

        justification = (
            parsed.get("justification") or
            parsed.get("reason") or
            parsed.get("explanation") or
            fallback["justification"]
        )

        product_area = parsed.get("product_area") or fallback["product_area"]

        return {
            "status": status,
            "product_area": product_area,
            "response": str(response_text),
            "justification": str(justification),
            "request_type": request_type,
        }

    # -------------------------
    # Helpers
    # -------------------------
    def _fallback(self):
        return {
            "status": "escalated",
            "product_area": "General Support",
            "response": "We could not process your request automatically. A support agent will assist you shortly.",
            "justification": "Model failed to produce a valid response.",
            "request_type": "product_issue",
        }

    def _format_passages(self, passages):
            if not passages:
                return ""
            
            formatted_chunks = []
            for i, p in enumerate(passages):
                source = p.get('source', 'Unknown Document')
                text = p.get('text', '')
                # Pass the full text, do not truncate with [:400]
                formatted_chunks.append(f"--- DOCUMENT {i+1}: {source} ---\n{text}")
                
            return "\n\n".join(formatted_chunks)
    def _read_csv(self, path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def _write_csv(self, path, rows):
        fields = [
            "issue", "subject", "company",
            "response", "product_area", "status",
            "request_type", "justification",
        ]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved → {path}")