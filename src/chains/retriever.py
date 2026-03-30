"""
src/chains/retriever.py
Loads the FAISS index and retrieves the top-k most relevant chunks
for a given query. Returns chunks with their metadata for citations.
"""

import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
INDEX_PATH   = BASE_DIR / "vector_store" / "index.faiss"
META_PATH    = BASE_DIR / "vector_store" / "index.pkl"
EMBED_MODEL  = "all-MiniLM-L6-v2"

# Load once at import time — shared across all queries
_model  = SentenceTransformer(EMBED_MODEL)
_index  = faiss.read_index(str(INDEX_PATH))
with open(META_PATH, "rb") as f:
    _chunks = pickle.load(f)


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Retrieve top-k chunks most similar to the query.

    Returns list of dicts:
    {
        "chunk_id": str,
        "text":     str,
        "source":   str,
        "section":  str,
        "score":    float,
    }
    """
    embedding = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(embedding)

    scores, indices = _index.search(embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def retrieve_multi(queries: list[str], k: int = 5) -> list[dict]:
    """
    Run multiple queries and deduplicate results by chunk_id.
    Useful when the planner needs evidence from multiple angles.
    """
    seen      = set()
    combined  = []

    for query in queries:
        for chunk in retrieve(query, k=k):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                combined.append(chunk)

    # Sort by score descending
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM.
    Includes source metadata so the planner can cite correctly.
    """
    parts = []
    for c in chunks:
        parts.append(
            f"[Source: {c['source']} | Section: {c['section']} | Score: {c['score']:.3f}]\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Quick test
    results = retrieve("CSE minor prerequisites eligibility CGPA", k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['source']} / {r['section']}")
        print(r["text"][:200])
        print()