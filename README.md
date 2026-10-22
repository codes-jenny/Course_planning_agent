# IIT KGP Course Planning Assistant

Agentic RAG system for ISE students planning a CSE Minor at IIT Kharagpur.

## Setup

Clone the repository and install dependencies:

```bash
git clone Course_planning_agent
cd iit-kgp-rag
pip install -r requirements.txt

Add your OpenAI key to .env:

OPENAI_API_KEY=sk-...

Run Order
Build the index (run once):
  python src/ingest.py
Test via CLI:
  python src/pipeline.py
Launch the UI:
  python app.py
Run evaluation:
  python eval/run_eval.py


Project Structure
data/raw/          ← Original PDFs
data/processed/    ← Cleaned JSONs + text files
src/               ← All pipeline code
vector_store/      ← Auto-generated FAISS index
eval/              ← Test queries + evaluation script
app.py             ← Gradio UI

Features
Prerequisite Checking: Multi-hop chains, co-requisites, either/or conditions, minimum grade enforcement.
Term Planning: Suggests courses based on completed courses, program requirements, and term limits.
Clarifying Questions: Requests missing info from students (e.g., grades, target term, catalog year).
Grounded Answers: Every response includes citations from the IIT KGP catalog documents.
Safe Abstention: If info is missing from catalog, provides guidance instead of guessing.
