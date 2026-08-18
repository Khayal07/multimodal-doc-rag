# Multimodal RAG Service

Production-grade Multimodal RAG system built with **FastAPI**, **LlamaParse**, **OpenAI**, and **ChromaDB**. Ingests complex PDFs (plain text, markdown tables, visual layouts), indexes them into a vector database, and answers questions with explicit per-page / per-table citations.

## Features

- **LlamaParse ingestion** — parses PDFs into structured, page-aware elements (text, headings, tables, lists).
- **Tables are never split** — each table is indexed whole and numbered per page (e.g. "Page 3, Table 1") for precise citations.
- **Citation-aware answers** — the LLM is constrained to cite numbered context slots; markers are post-processed into readable `[Source: Page X, Table Y]` labels, and a structured `sources` list is returned alongside every answer.
- **Built-in test UI** — a lightweight single-page UI at `GET /` for PDF upload, questions, and citation-highlighted answers (no build step).
- **Zero hardcoding** — every model name, path, collection, and credential is resolved from `.env` via `pydantic-settings`.
- **Pluggable vector store** — a `BaseVectorStore` interface with a ChromaDB implementation; Qdrant can be added behind the same contract.
- **Resilience** — exponential backoff retries on external API rate limits and transient failures.

## Architecture

```
POST /api/v1/ingest/pdf (PDF)                    POST /api/v1/query
        │                                                │
        ▼                                                ▼
  LlamaParse (markdown, page-aware elements)     EmbeddingService (query → vector)
        │                                                │
        ▼                                                ▼
  Chunker (text split + tables kept whole)         Retriever → VectorStore.query
        │                                                │
        ▼                                                ▼
  EmbeddingService (chunks → vectors)              GenerationPipeline (context → LLM)
        │                                                │
        ▼                                                ▼
  VectorStore.add (ChromaDB + metadata)            resolve_citations → answer + sources
```

Key modules under `app/`:

| Module | Responsibility |
| --- | --- |
| `ingestion/parser.py` | Async LlamaParse wrapper (markdown mode); produces `ParsedDocument` with page + element metadata |
| `ingestion/chunker.py` | Converts elements into citation-aware `Chunk`s; tables kept whole with per-page `table_id` |
| `ingestion/loader.py` | Orchestrates parse → chunk → embed → store; re-ingest replaces prior version |
| `embeddings/service.py` | OpenAI embedding client with batched, retry-safe calls |
| `vectorstore/` | `BaseVectorStore` interface, `ChromaStore` implementation, backend factory |
| `retrieval/retriever.py` | Embeds the query and returns the top-k nearest chunks |
| `generation/pipeline.py` | Retrieval + prompt assembly + LLM call + citation resolution |
| `generation/prompt.py` | System prompt, numbered context formatting, `[n] → [Source: ...]` rewriting |
| `config.py` | Typed settings loaded from `.env` (no hardcoded values) |

## Getting started

### 1. Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS / Linux
.venv/bin/pip install -r requirements.txt

copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```

Fill in your credentials in `.env`:

```ini
OPENAI_API_KEY=sk-...
LLAMA_CLOUD_API_KEY=llx-...   # LlamaParse (free tier: 1000 pages/day)
```

### 2. Run the service

```bash
.venv\Scripts\uvicorn app.main:app --reload
# or
.venv\Scripts\python -m uvicorn app.main:app --reload
```

Interactive docs: http://localhost:8000/docs
Test UI: http://localhost:8000 (upload a PDF, ask a question, see cited answers)

### 3. Ingest a PDF

```bash
curl -X POST http://localhost:8000/api/v1/ingest/pdf \
  -F "file=@report.pdf;type=application/pdf"
```

Response:

```json
{
  "document": "report.pdf",
  "pages": 12,
  "element_counts": {"text": 84, "table": 5, "heading": 18},
  "chunks_indexed": 61,
  "status": "indexed"
}
```

### 4. Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Q3 revenue per region?", "top_k": 5}'
```

Response includes the cited answer plus structured sources:

```json
{
  "query": "What was Q3 revenue per region?",
  "answer": "Q3 revenue was ... [Source: Page 3, Table 1] ... [Source: Page 7]",
  "sources": [
    {
      "chunk_id": "ab12cd34ef56-p3-14",
      "document": "report.pdf",
      "page": 3,
      "element_type": "table",
      "table_id": 1,
      "text": "| Region | Q3 |\n|---...",
      "score": 0.12
    }
  ],
  "citations": ["[Source: Page 3, Table 1]", "[Source: Page 7]"]
}
```

## Configuration

All values live in `.env` (see `.env.example` for the complete list with defaults).

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | OpenAI credentials |
| `LLAMA_CLOUD_API_KEY` | — | LlamaParse credentials |
| `LLM_MODEL` | `gpt-4o-mini` | Answer generation model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Chunk/query vectorization model |
| `VECTOR_DB_TYPE` | `chroma` | `chroma` (local) or `qdrant` (cloud) |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `multimodal_docs` | Collection / namespace |
| `CHUNK_SIZE` | `1500` | Target chunk length (characters) |
| `CHUNK_OVERLAP` | `150` | Overlap between text chunks |
| `TOP_K` | `5` | Default chunks retrieved per query |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature for answers |
| `MAX_RETRIES` | `3` | Retry count for external APIs |

## Development

```bash
# Run tests (mocked, no API keys or network required)
.venv\Scripts\python -m pytest -q

# Lint
.venv\Scripts\python -m ruff check app tests scripts

# Health check
curl http://localhost:8000/health
```

## Notes & limitations

- LlamaParse free tier is limited to **1000 pages/day**; parse calls are retried with exponential backoff on rate limits.
- Tables are indexed whole and never split, so a single very large table becomes one chunk.
- Image elements are skipped during ingestion (their captions, if any, are not separately extracted).
- The `qdrant` backend is selectable via `VECTOR_DB_TYPE` but not yet implemented; only `chroma` is supported today.