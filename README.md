# EBooksSorter

Local-first PDF/EPUB library explorer with grounded RAG Q&A and semantic chunk indexing.

## What This Project Does

- Scans a source folder of `.pdf` and `.epub` files and categorizes books.
- Builds semantic artifacts for search and RAG:
  - `output/semantic_source.jsonl`
  - `output/semantic_chunks.jsonl`
  - `output/semantic_index/`
  - `output/semantic_index_chunks/`
- Runs a Streamlit dashboard (`dashboard.py`) for:
  - semantic search,
  - Ask Books grounded Q&A with citations,
  - recommendation and graph views.

## Requirements

- macOS/Linux
- Python 3.10+ (project currently runs with `.venv`)
- Optional:
  - Ollama (for text generation backend)

## Quick Start

### 1) Install dependencies

```bash
cd ~/Projects/EBooksSorter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build source records

```bash
.venv/bin/python index_books.py \
  --config "./categories.yaml" \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output"
```

### 3) Build semantic index (book-level)

```bash
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_source.jsonl" \
  --output-dir "./output/semantic_index" \
  --model "sentence-transformers/all-MiniLM-L6-v2"
```

### 4) Build chunk index (RAG)

```bash
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks" \
  --model "sentence-transformers/all-MiniLM-L6-v2"
```

#### Compare multiple text embedding models (optional)

```bash
# MiniLM (faster, lighter)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_minilm" \
  --model "sentence-transformers/all-MiniLM-L6-v2"

# BGE base (usually better retrieval quality)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_bge_base" \
  --model "BAAI/bge-base-en-v1.5"

# BGE large (higher quality, slower/heavier)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_bge_large" \
  --model "BAAI/bge-large-en-v1.5"

# MXBAI large (strong retrieval quality)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_mxbai_large" \
  --model "mixedbread-ai/mxbai-embed-large-v1"

# GTE large (quality-focused alternative)
.venv/bin/python build_semantic_index.py \
  --semantic-source "./output/semantic_chunks.jsonl" \
  --output-dir "./output/semantic_index_chunks_gte_large" \
  --model "thenlper/gte-large"
```

### 5) Start API (optional but recommended for API mode)

```bash
export RAG_API_KEY="change-this-internal-key"
.venv/bin/python manage.py runserver 0.0.0.0:8000 --noreload
```

### 6) Start Streamlit dashboard

```bash
.venv/bin/streamlit run dashboard.py
```

## Optional Backends

### Ollama (text generation)

```bash
ollama pull granite3.3:8b
ollama run granite3.3:8b "hello"
```

Use in Ask Books:

- generation mode: `ollama`
- base URL: `http://127.0.0.1:11434`
- model: `granite3.3:8b`

## Ask Books: Recommended Settings

### Grounded text retrieval

- reranker: enabled
- fallback: enabled

## API Smoke Test

```bash
curl -X POST "http://127.0.0.1:8000/rag/answer" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${RAG_API_KEY}" \
  -d '{
    "query": "Explain a neural network using grounded citations",
    "top_k": 4,
    "max_citations": 3,
    "ollama": {
      "enabled": true,
      "base_url": "http://127.0.0.1:11434",
      "model": "granite3.3:8b"
    }
  }'
```

## Outputs You Should See

- `output/semantic_index/*.npy|*.json`
- `output/semantic_index_chunks/*.npy|*.json`

## Troubleshooting

- **`/rag/*` returns 401/503**: set `RAG_API_KEY` on server and send `X-API-Key` header.
- **Slow first run**: embedding models download on first use.
- **No chunk index**: rebuild with `--semantic-source output/semantic_chunks.jsonl`.

## Document Sync Notes

- `RUNBOOK.md` and `DEPLOYMENT.md` are aligned with this README:
  - index refresh remains `index_books.py` + `build_semantic_index.py`.
