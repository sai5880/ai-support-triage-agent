"""
scrape_corpus.py — Optional helper to populate data/ from support sites.

Usage:
  pip install requests beautifulsoup4
  python scrape_corpus.py

This script crawls the three support domains and saves text to data/.
Run this ONCE to build the corpus, then use main.py for triage.

NOTE: Respect robots.txt and rate limits. This is for the hackathon corpus only.
"""

import os
import time
import re
import urllib.parse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Install deps: pip install requests beautifulsoup4")
    raise

SITES = {
    "hackerrank": {
        "start_urls": [
            "https://support.hackerrank.com/hc/en-us",
        ],
        "domain": "support.hackerrank.com",
    },
    "claude": {
        "start_urls": [
            "https://support.claude.ai/hc/en-us",
        ],
        "domain": "support.claude.ai",
    },
    "visa": {
        "start_urls": [
            "https://www.visa.co.in/support.html",
        ],
        "domain": "visa.co.in",
    },
}

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_PAGES_PER_SITE = 150
DELAY = 1.0  # seconds between requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; support-corpus-builder/1.0; hackathon)"
}


def clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if len(l) > 20]
    return "\n".join(lines)


def crawl_site(name: str, config: dict):
    out_dir = os.path.join(OUTPUT_BASE, name)
    os.makedirs(out_dir, exist_ok=True)

    visited = set()
    queue = list(config["start_urls"])
    domain = config["domain"]
    saved = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    while queue and saved < MAX_PAGES_PER_SITE:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract and save text
            text = clean_text(soup)
            if len(text) > 100:
                slug = re.sub(r"[^a-z0-9]+", "_", url.lower())[-80:]
                fname = os.path.join(out_dir, f"{slug}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(f"SOURCE: {url}\n\n{text}")
                saved += 1
                print(f"  [{name}] Saved {saved}: {url[:80]}")

            # Enqueue internal links
            for a in soup.find_all("a", href=True):
                href = a["href"]
                full = urllib.parse.urljoin(url, href)
                parsed = urllib.parse.urlparse(full)
                if domain in parsed.netloc and full not in visited:
                    queue.append(full)

        except Exception as e:
            print(f"  [{name}] Error on {url}: {e}")

        time.sleep(DELAY)

    print(f"  [{name}] Done. Saved {saved} pages.")


if __name__ == "__main__":
    for name, config in SITES.items():
        print(f"\nCrawling {name}...")
        crawl_site(name, config)
    print("\nCorpus build complete.")