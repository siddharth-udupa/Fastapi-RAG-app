# RAG App — FastAPI + pgvector + Gemini

A small, readable Retrieval-Augmented Generation system built for learning.
It is intentionally split into small modules so each RAG concept can be
replaced or extended without rewriting the app.

---

## What it does

1. You upload a PDF.
2. The PDF is parsed, split into chunks, embedded, and stored in PostgreSQL
   (with the `pgvector` extension).
3. You ask a question.
4. The question is embedded, the closest chunks are retrieved, and the result
   is sent to Gemini together with the question.
5. The answer is returned with the sources that were used.

**Important:** the LLM is called whether or not retrieval finds anything.
When there are no matches, a different system prompt tells the model to say so
and answer from general knowledge instead of failing with a 404.

---

## The two flows

### Ingestion

```
PDF bytes
   │  app/services/ingestion.py :: parse_pdf()
   ▼
raw text
   │  app/services/chunking.py :: chunk_text()
   ▼
chunks (list[str])
   │  app/services/embedding.py :: embed_documents()   ← one batched API call
   ▼
vectors (3072 floats each)
   │  SQLAlchemy
   ▼
documents + document_chunks tables
```

### Query

```
question
   │  app/services/embedding.py :: embed_query()       ← RETRIEVAL_QUERY task
   ▼
query vector
   │  app/services/retrieval.py :: retrieve_chunks()   ← pgvector cosine search
   ▼
RagContext (0..top_k chunks)
   │  app/services/prompts.py :: build_user_prompt()   ← context construction
   ▼
prompt + system instruction
   │  app/services/generation.py
   ▼
Gemini answer (streamed or complete)
```

The query flow is orchestrated by `RagPipeline` in
`app/services/rag.py`. That file is the best place to start reading.

---

## Project layout

```
main.py                      FastAPI app, /health, static frontend mount
app/
  core/
    config.py                Settings loaded from .env (one source of truth)
    db.py                    Engine, session factory, get_db dependency
    llm.py                   One shared Gemini client (lru_cache)
  models/
    document.py              Document (1) ── DocumentChunk (many, with vector)
  schemas/
    document.py              Upload request/response shapes
    query.py                 Query request/response shapes
  services/
    chunking.py              Text splitting
    embedding.py             Embedding calls (documents vs queries)
    ingestion.py             PDF -> chunks -> embeddings -> DB
    retrieval.py             Vector search + RetrievedChunk dataclass
    prompts.py               System prompts + context formatting
    generation.py            Non-streaming and streaming LLM calls
    rag.py                   RagPipeline: prepare() / answer() / stream()
  api/
    documents.py             POST /documents/upload
    query.py                 POST /query, POST /query/stream (SSE)
    router.py                Route wiring
  static/index.html          Small frontend using both query endpoints
alembic/                     Database migrations
```

---

## API

| Method | Path                        | Purpose                              |
| ------ | --------------------------- | ------------------------------------ |
| POST   | `/api/v1/documents/upload`  | Upload and ingest a PDF              |
| POST   | `/api/v1/query`             | Full JSON answer + sources           |
| POST   | `/api/v1/query/stream`      | Streamed answer + sources (SSE)      |
| GET    | `/health`                   | Health check                         |

```bash
# Upload
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@notes.pdf"

# Ask (JSON)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main conclusions?", "top_k": 5}'

# Ask (streaming)
curl -N -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main conclusions?"}'
```

### SSE event protocol

`/api/v1/query/stream` sends:

```
event: sources
data: {"sources": [...], "used_context": true}

event: token
data: {"text": "The document says ..."}

event: done
data: {"model": "gemini-3.8-flash"}
```

If the LLM fails after the stream has started, an `error` event is sent
(the HTTP status cannot change at that point).

Why SSE instead of plain text? Sources arrive before the first token, so the
UI can show citations immediately, and new event types can be added later
without breaking clients.

---

## Setup

```bash
cd FastAPI/RAG-app

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your DATABASE_URL and GEMINI_API_KEY

alembic upgrade head
uvicorn main:app --reload
```

Open <http://localhost:8000> for the small UI, or
<http://localhost:8000/docs> for the interactive API docs.

---

## The extension points (for advanced RAG)

### 1. Swap the retriever

`RagPipeline` accepts any object matching the `Retriever` protocol in
`app/services/rag.py`:

```python
def retrieve_chunks(
    *, question: str, db: Session, top_k: int, min_similarity: float
) -> list[RetrievedChunk]: ...
```

A hybrid retriever only has to return the same `list[RetrievedChunk]`:

```python
class HybridRetriever:
    def __call__(self, *, question, db, top_k, min_similarity):
        vector_hits = retrieve_chunks(...)
        keyword_hits = search_bm25(...)
        return merge_and_dedupe(vector_hits, keyword_hits)[:top_k]

pipeline = RagPipeline(retriever=HybridRetriever())
```

The API and generation layers do not change.

In the API, `get_pipeline()` in `app/api/query.py` is a FastAPI dependency, so
you can switch retrievers in one place or override it in tests:

```python
# app/api/query.py
def get_pipeline() -> RagPipeline:
    return RagPipeline(retriever=HybridRetriever())
```

```python
# tests
app.dependency_overrides[get_pipeline] = lambda: RagPipeline(retriever=FakeRetriever())
```

### 2. Add stages inside `RagPipeline.prepare()`

`prepare()` is the one place where the question becomes a `RagContext`.
Add advanced stages there, in order:

| Concept                     | Where it fits                              |
| --------------------------- | ------------------------------------------ |
| Query rewriting / expansion | before calling the retriever               |
| Multi-query retrieval       | call the retriever several times, merge    |
| Hybrid search (BM25 + vec)  | a new `Retriever` implementation           |
| Metadata filtering          | add filters to the retriever protocol      |
| Reranking (cross-encoder)   | reorder `chunks` after retrieval           |
| MMR / deduplication         | post-process `chunks`                      |
| Context compression         | summarise/prune chunks before prompting    |
| Parent-document retrieval   | fetch parent text using chunk metadata     |
| Caching                     | wrap `prepare()` or the retriever          |
| Evaluation                  | consume `RagContext` / `RagAnswer` directly |

Because `RagContext` and `RagAnswer` are plain dataclasses, you can log them,
store them, or score them without invoking the LLM again.

### 3. Change prompts independently

All prompts live in `app/services/prompts.py`. Compare prompt versions without
touching retrieval or API code.

### 4. Add conversation memory

`generation.py` currently sends one user message. Add prior turns to
`contents` there; the retriever can optionally rewrite follow-up questions
into standalone questions first.

---

## Design decisions worth knowing

**Sync endpoints, not `async def`.** The Gemini SDK and SQLAlchemy calls are
blocking. FastAPI runs `def` endpoints in a worker thread, so the event loop
stays free. Using `async def` with blocking clients would stall every request.

**One shared Gemini client.** `app/core/llm.py` caches the client with
`lru_cache`. Creating a new client per request wastes connection setup.

**No context is not an error.** `retrieve_chunks()` returns `[]` for an empty
database. `RagPipeline` passes that to the LLM with a no-context system prompt.
The API returns `used_context: false` and an empty `sources` list. If you prefer
a strict refusal instead of a general-knowledge answer, edit
`SYSTEM_PROMPT_WITHOUT_CONTEXT` in `app/services/prompts.py`.

**Embedding dimension matters.** `gemini-embedding-2` returns 3072 floats, so
the column is `vector(3072)` (see the second Alembic migration). If you change
`EMBEDDING_MODEL`, check its output size and create a migration that resizes
the column.

**No vector index yet.** pgvector's HNSW/IVFFlat indexes support at most 2000
dimensions for the `vector` type. With 3072 dimensions, Postgres does an exact
scan, which is fine for a learning project. For large collections, store
`halfvec(3072)` or reduce the embedding size, then add an index.

**Secrets stay out of git.** `alembic/env.py` reads `DATABASE_URL` from
`app.core.config`, so `alembic.ini` contains no credentials. `.env` is ignored.

**Alembic ignores unrelated tables.** `include_object` in `alembic/env.py`
prevents `--autogenerate` from dropping tables that belong to other projects
in the same database.
