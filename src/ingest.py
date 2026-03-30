"""
ingest.py
Reads all files from data/processed/ and data/raw/minor_cse.pdf,
chunks them, embeds with sentence-transformers, builds a FAISS index.

Run once before anything else:
    python src/ingest.py
"""

import os
import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
RAW_DIR         = BASE_DIR / "data" / "raw"
VECTOR_STORE    = BASE_DIR / "vector_store"
INDEX_PATH      = VECTOR_STORE / "index.faiss"
META_PATH       = VECTOR_STORE / "index.pkl"

VECTOR_STORE.mkdir(exist_ok=True)

# ── Embedding model (local, no API key needed) ───────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good for retrieval

# ── Chunking config ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # tokens (~400 words) — large enough to hold a full prereq rule
CHUNK_OVERLAP = 50    # keeps context across chunk boundaries


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_json_courses(filepath: Path) -> list[dict]:
    """
    Convert each course entry into a rich text document.
    Each course becomes ONE document so prereq info stays together.
    """
    with open(filepath) as f:
        courses = json.load(f)

    docs = []
    for c in courses:
        prereqs = ", ".join(c["prerequisite"]) if c["prerequisite"] else "None"
        text = (
            f"Course Code: {c['course_code']}\n"
            f"Course Name: {c['course_name']}\n"
            f"Credits: {c['credits']}\n"
            f"Prerequisites: {prereqs}\n"
            f"Description: {c['description']}\n"
            f"Source: {c.get('source', 'IIT Kharagpur')}"
        )
        docs.append({
            "text":    text,
            "source":  str(filepath.name),
            "section": c["course_code"],
            "type":    "course",
        })
    return docs


def load_text_file(filepath: Path) -> list[dict]:
    """
    Load a plain .txt file. Returns as a single document;
    the splitter will chunk it below.
    """
    text = filepath.read_text(encoding="utf-8")
    return [{
        "text":    text,
        "source":  filepath.name,
        "section": filepath.stem,
        "type":    "policy",
    }]


def load_pdf(filepath: Path) -> list[dict]:
    """
    Extract text from every page of a PDF.
    Each page becomes one document; splitter chunks further if needed.
    """
    reader = PdfReader(str(filepath))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append({
                "text":    text,
                "source":  filepath.name,
                "section": f"page_{i+1}",
                "type":    "policy" if "policy" in filepath.stem.lower() else "program",
            })
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split long documents into chunks.
    JSON course docs are short enough to skip splitting (kept whole).
    Text/PDF docs are split with overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = []
    for doc in docs:
        if doc["type"] == "course":
            # Course entries are short — keep whole, no splitting
            chunks.append({
                "chunk_id": f"{doc['section']}_0",
                "text":     doc["text"],
                "source":   doc["source"],
                "section":  doc["section"],
                "type":     doc["type"],
            })
        else:
            # Split policy/program text into overlapping chunks
            parts = splitter.split_text(doc["text"])
            for i, part in enumerate(parts):
                chunks.append({
                    "chunk_id": f"{doc['section']}_chunk_{i}",
                    "text":     part,
                    "source":   doc["source"],
                    "section":  doc["section"],
                    "type":     doc["type"],
                })
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDER + INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_index(chunks: list[dict]):
    """
    Embed all chunks and write FAISS index + metadata pickle to vector_store/.
    """
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # L2-normalise so cosine similarity == dot product (works with IndexFlatIP)
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product = cosine after normalisation
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index saved   → {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")
    print(f"Metadata saved → {META_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_docs: list[dict] = []

    # ── 1. JSON course files ─────────────────────────────────────────────────
    for json_file in PROCESSED_DIR.glob("*.json"):
        print(f"Loading JSON : {json_file.name}")
        all_docs.extend(load_json_courses(json_file))

    # ── 2. Text files ────────────────────────────────────────────────────────
    for txt_file in PROCESSED_DIR.glob("*.txt"):
        print(f"Loading TXT  : {txt_file.name}")
        all_docs.extend(load_text_file(txt_file))

    # ── 3. PDFs from processed/ (ISE brochure etc.) ──────────────────────────
    for pdf_file in PROCESSED_DIR.glob("*.pdf"):
        print(f"Loading PDF  : {pdf_file.name}")
        all_docs.extend(load_pdf(pdf_file))

    # ── 4. minor_cse.pdf from raw/ ───────────────────────────────────────────
    minor_pdf = RAW_DIR / "minor_cse.pdf"
    if minor_pdf.exists():
        print(f"Loading PDF  : {minor_pdf.name}  (from raw/)")
        all_docs.extend(load_pdf(minor_pdf))
    else:
        print(f"WARNING: {minor_pdf} not found — CSE minor requirements will be missing!")

    # ── 5. Chunk + embed + index ─────────────────────────────────────────────
    print(f"\nTotal documents loaded : {len(all_docs)}")
    chunks = chunk_documents(all_docs)
    print(f"Total chunks after split: {len(chunks)}")

    build_index(chunks)
    print("\nIngestion complete. Run pipeline.py next.")


if __name__ == "__main__":
    main()