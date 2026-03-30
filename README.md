# IIT KGP Course Planning Assistant
Agentic RAG system for ISE students planning a CSE Minor at IIT Kharagpur.

## Setup

```bash
git clone <your-repo>
cd iit-kgp-rag
pip install -r requirements.txt
```

Add your OpenAI key to `.env`:
```
OPENAI_API_KEY=sk-...
```

## Run order

### 1. Build the index (run once)
```bash
python src/ingest.py
```

### 2. Test via CLI
```bash
python src/pipeline.py
```

### 3. Launch UI
```bash
python app.py
```

### 4. Run evaluation
```bash
python eval/run_eval.py
```

## Project structure
```
data/raw/          ← original PDFs
data/processed/    ← cleaned JSONs + text files
src/               ← all pipeline code
vector_store/      ← auto-generated FAISS index
eval/              ← test queries + evaluation script
app.py             ← Gradio UI
```