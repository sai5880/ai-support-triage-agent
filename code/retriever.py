import os
import re
import pickle
import numpy as np
import faiss
from typing import List, Dict
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
CHUNK_SIZE_CHARS = 2000
OVERLAP_CHARS = 250
BATCH_SIZE = 32
INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "chunks_metadata.pkl"

# ------------------------------------------------------------------ #
# Utils
# ------------------------------------------------------------------ #

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_matrix(x: np.ndarray):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


# ------------------------------------------------------------------ #
# Retriever (Optimized with FAISS, strict I/O compliance)
# ------------------------------------------------------------------ #

class Retriever:
    def __init__(self, corpus_dir: str, debug: bool = True):
            self.corpus_dir = corpus_dir
            self.debug = debug
            self.chunks: List[Dict] = []
            self.index = None

            self._log("Initializing Retriever...")

            # --- OPEN VINO INTEL GPU LOAD ---
            self._log("Loading BGE-Small into Intel Iris Xe via OpenVINO...")
            model_id = "BAAI/bge-small-en-v1.5"

            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = OVModelForFeatureExtraction.from_pretrained(model_id, export=True, device="GPU")

            # Load FAISS index if it exists, otherwise build it
            if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
                self._load_index()
            else:
                self._build_embeddings()

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _log(self, *args):
        if not self.debug:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}]", *args)

    # ------------------------------------------------------------------ #
    # Embeddings (FAISS Indexing)
    # ------------------------------------------------------------------ #

    def _build_embeddings(self):
            self._log("Building fresh index from files... (This happens ONCE)")
            self.chunks = []
            
            # 1. Chunking
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE_CHARS,
                chunk_overlap=OVERLAP_CHARS,
                length_function=len,
                separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
            )
    
            total_files = 0
            for root, _, files in os.walk(self.corpus_dir):
                for fname in files:
                    if not fname.lower().endswith((".md", ".txt", ".html", ".json")):
                        continue
                    
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        
                        rel_path = os.path.relpath(fpath, self.corpus_dir)
                        parts = rel_path.split(os.sep)
                        domain = parts[0] if len(parts) > 1 else "general"
    
                        texts = splitter.split_text(clean_text(text))
                        for t in texts:
                            if len(t.strip()) > 50:
                                self.chunks.append({
                                    "text": t.strip(),
                                    "source": rel_path,
                                    "domain": domain.lower()
                                })
                        total_files += 1
                    except Exception as e:
                        self._log(f"[WARN] Failed reading {fpath}: {e}")
    
            self._log(f"Corpus loaded: {total_files} files → {len(self.chunks)} chunks")
    
            if not self.chunks:
                self._log("[WARN] No data found to index.")
                return
    
            # 2. Embedding using OpenVINO on GPU
            texts_to_embed = [c["text"] for c in self.chunks]
            self._log(f"Encoding {len(texts_to_embed)} chunks on Iris Xe GPU...")
            
            all_embeddings = []
            
            for i in range(0, len(texts_to_embed), BATCH_SIZE):
                batch_texts = texts_to_embed[i:i + BATCH_SIZE]
    
                # Tokenize the text batch
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                
                # Forward Pass through Iris Xe GPU
                outputs = self.model(**inputs)
                
                # Extract the CLS token (index 0) for sentence embedding and convert to numpy
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                all_embeddings.append(batch_embeddings)
    
                if self.debug and (i // BATCH_SIZE) % 5 == 0:
                    self._log(f"Processed {min(i + BATCH_SIZE, len(texts_to_embed))}/{len(texts_to_embed)} chunks...")
    
            # Combine all batches into one matrix
            embeddings = np.vstack(all_embeddings)
            embeddings = normalize_matrix(embeddings)
    
            # 3. Save to FAISS
            self.index = faiss.IndexFlatIP(embeddings.shape[1]) 
            self.index.add(embeddings)
    
            faiss.write_index(self.index, INDEX_FILE)
            import pickle
            with open(METADATA_FILE, "wb") as f:
                pickle.dump(self.chunks, f)
                
            self._log("Index built and saved successfully!")

    def _load_index(self):
        self._log("Loading pre-built FAISS index from disk...")
        self.index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            self.chunks = pickle.load(f)
        self._log(f"Loaded {len(self.chunks)} chunks instantly.")


    # ------------------------------------------------------------------ #
    # Retrieval (Strict I/O compliance)
    # ------------------------------------------------------------------ #

    
    def retrieve(self, issue: str, subject: str, company: str, top_k: int = 7) -> List[Dict]:
            if not self.index or not self.chunks:
                return []

            query_text = f"{subject}\n{issue}".strip()
            self._log(f"\n[QUERY] {query_text}")

            # Tokenize and embed the single query using OpenVINO
            inputs = self.tokenizer([query_text], padding=True, truncation=True, max_length=512, return_tensors="pt")
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].numpy()

            query_embedding = normalize_matrix(query_embedding)

            # ---------------- FAST PASS (FAISS) ----------------
            search_k = max(top_k * 3, 21) 
            distances, indices = self.index.search(query_embedding, search_k)

            # ---------------- REFINE & BOOST ----------------
            refined_scores = []
            company = (company or "").lower()

            for score, idx in zip(distances[0], indices[0]):
                if idx == -1: continue

                chunk = self.chunks[idx]
                final_score = float(score)

                if company and company in chunk["domain"]:
                    final_score *= 1.5 

                refined_scores.append((final_score, idx))

            refined_scores.sort(key=lambda x: x[0], reverse=True)

            # ---------------- FINAL OUTPUT ----------------
            results = []
            seen_sources = set()

            for score, idx in refined_scores:
                chunk = self.chunks[idx]

                # Original formatting: text and source ONLY.
                if chunk["source"] not in seen_sources:
                    seen_sources.add(chunk["source"])
                    results.append({
                        "text": chunk["text"],
                        "source": chunk["source"]
                    })

                if len(results) >= top_k:
                    break
                
            return results
