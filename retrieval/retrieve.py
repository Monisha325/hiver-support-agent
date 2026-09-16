"""
Phase 2 — Retrieval logic.

Given a customer query string, returns the top-k most similar
AmazonHelp historical resolutions from the FAISS index.

Grounding guarantee: every result comes from a real AmazonHelp reply
in twcs.csv. No model pretrained knowledge is used here.

Usage (from agent code):
    from retrieval.retrieve import Retriever
    r = Retriever()
    hits = r.query("my package hasn't arrived", top_k=5)
    # hits: list of dicts with customer_text, reply_text, score, intent_seed
"""

import json
import pathlib
import re

import numpy as np

ROOT          = pathlib.Path(__file__).parent.parent
RETRIEVAL_DIR = pathlib.Path(__file__).parent
INDEX_PATH    = RETRIEVAL_DIR / "index.faiss"
CORPUS_PATH   = RETRIEVAL_DIR / "corpus.jsonl"
MODEL_NAME    = "all-MiniLM-L6-v2"


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Retriever:
    """
    Thin wrapper around the FAISS index + corpus JSONL.
    Loads everything lazily on first call to query().
    """

    def __init__(self):
        self._index   = None
        self._corpus  = None
        self._model   = None

    def _load(self):
        if self._index is not None:
            return

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run: python retrieval/build_index.py"
            )

        import faiss
        from sentence_transformers import SentenceTransformer

        self._index  = faiss.read_index(str(INDEX_PATH))
        self._model  = SentenceTransformer(MODEL_NAME)
        self._corpus = []
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                self._corpus.append(json.loads(line))

    def query(self, customer_text: str, top_k: int = 5) -> list[dict]:
        """
        Returns top_k most similar resolved threads.

        Each result dict contains:
          idx           — corpus position
          customer_text — original customer tweet from twcs.csv
          reply_text    — AmazonHelp's actual reply from twcs.csv
          intent_seed   — seed-labelled intent category
          score         — cosine similarity (higher = more similar)

        GROUNDING NOTE: reply_text is always verbatim from twcs.csv.
        The agent may use it as-is or as a template for drafting — but the
        source is always the brand's own historical reply, never the LLM's
        pretrained knowledge alone.
        """
        self._load()

        cleaned = _clean(customer_text)
        vec = self._model.encode(
            [cleaned],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self._index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:   # FAISS sentinel for "not enough results"
                continue
            hit = dict(self._corpus[idx])
            hit["score"] = float(score)
            results.append(hit)
        return results


# ── quick smoke test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    r = Retriever()
    test_queries = [
        "Where is my order? It was supposed to arrive yesterday",
        "I need to return a broken item and get a refund",
        "I can't log into my account, password reset email not working",
        "My Amazon Echo keeps shutting down",
        "I was charged twice for the same order",
    ]
    for q in test_queries:
        print(f"\nQUERY: {q}")
        hits = r.query(q, top_k=3)
        for i, h in enumerate(hits, 1):
            print(f"  [{i}] score={h['score']:.3f} | intent={h['intent_seed']}")
            print(f"       customer: {h['customer_text'][:100]}")
            print(f"       reply:    {h['reply_text'][:100]}")
