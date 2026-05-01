# Internal Knowledge Chat API

A RAG (Retrieval-Augmented Generation) API: upload documents, search by meaning, and chat with your knowledge base — every answer includes source references so claims can be verified.

---

## How to run

### Prerequisites
- Python 3.12+
- Git

### 1. Clone and create the virtual environment

```bash
git clone https://github.com/NimaaaAI/Internal-Knowledge-Chat-API.git
cd Internal-Knowledge-Chat-API
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install PostgreSQL 16

PostgreSQL is a system service — it lives outside the venv.

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
sudo service postgresql start
```

### 3. Unlock local access (dev only)

By default PostgreSQL uses `peer` auth on the Unix socket (must be the postgres OS user). Switch to `trust` for local dev:

```bash
sudo bash -c "sed -i 's/local   all             postgres                                peer/local   all             postgres                                trust/' /etc/postgresql/16/main/pg_hba.conf"
sudo service postgresql restart
```

### 4. Create the database and user

```bash
psql -U postgres
```

Inside the shell:

```sql
CREATE USER rag WITH PASSWORD 'rag';
CREATE DATABASE knowledge OWNER rag;
GRANT ALL PRIVILEGES ON DATABASE knowledge TO rag;
\q
```

### 5. Install pgvector

pgvector is a PostgreSQL extension (not a separate service). It adds vector storage and similarity search directly inside PostgreSQL.

```bash
cd /tmp
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
cd -
```

### 6. Apply the database schema

```bash
psql -U postgres -d knowledge -f init.sql
```

### 7. Set up environment variables

```bash
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY
```

### 8. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 9. Run the API

```bash
uvicorn app.main:app --reload
```

- API: `http://localhost:8000`
- UI: `http://localhost:8000` (open in browser)
- Interactive docs: `http://localhost:8000/docs`

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /text | Upload plain text with metadata |
| POST | /document | Upload a PDF or .txt file |
| GET | /search | Hybrid semantic + keyword search |
| POST | /chat | Ask a question, get answer + source references |
| GET | /documents | List all uploaded documents |
| DELETE | /documents/{id} | Delete a document and all its chunks |

---

## Architectural choices and tradeoffs

### Stack

| Layer | Choice | Reason |
|---|---|---|
| Framework | Python + FastAPI | Async-native, auto OpenAPI docs, fast to iterate |
| Database | PostgreSQL 16 + pgvector | One service handles relational data, vector search (HNSW), and full-text search (GIN) — no separate vector DB needed |
| Embeddings | `all-MiniLM-L6-v2` | 384-dim, free, no second API key, runs on CPU in ~80 ms/chunk |
| LLM | Claude claude-sonnet-4-6 (Anthropic) | Strong instruction-following, honest about uncertainty, good inline citations |
| PDF extraction | PyMuPDF | Fast, layout-aware, handles multi-column documents |

### Why pgvector instead of a dedicated vector DB (Qdrant, Weaviate)?

A dedicated vector DB requires running two services: one for relational data and one for vectors. With pgvector, one PostgreSQL instance handles everything — vector similarity search, full-text search, relational joins, and metadata filtering — all in a single SQL query. At knowledge-base scale (< 1M chunks), pgvector with HNSW is fast enough, and the SQL composability is worth more than the marginal performance gap.

If the corpus grew to tens of millions of chunks and sub-10 ms p99 latency were a hard requirement, Qdrant or Weaviate would be the right call.

### Data model

```
documents
  id            UUID        primary key
  title         TEXT        human-readable name
  source        TEXT        where it came from (URL, internal, etc.)
  author        TEXT        who wrote it
  doc_type      TEXT        memo | report | article | ...
  content_type  TEXT        'text' | 'document' (PDF)
  extra_metadata JSONB      arbitrary key-value pairs for filtering
  created_at    TIMESTAMPTZ

chunks
  id            UUID        primary key
  document_id   UUID        FK → documents.id (CASCADE DELETE)
  chunk_index   INTEGER     position within the document
  text          TEXT        raw chunk text
  embedding     vector(384) L2-normalised embedding for cosine search
  ts_vector     TSVECTOR    auto-maintained by DB trigger for FTS
  created_at    TIMESTAMPTZ
```

Indexes:
- `HNSW` on `chunks.embedding` — approximate nearest-neighbour vector search (better than IVFFlat for dynamic insert patterns)
- `GIN` on `chunks.ts_vector` — full-text keyword search
- B-tree on `chunks.document_id` — chunk lookup by document
- `GIN` on `documents.extra_metadata` — fast JSONB key-value filtering

### Chunk size: 150 words, 30-word overlap

`all-MiniLM-L6-v2` has a hard limit of 256 wordpiece tokens. English text averages ~1.4 wordpieces per word, so 150 words ≈ 210 wordpieces — safely under the cap with headroom for subword-heavy content.

- **Why not smaller (50 words)?** Too little context — the embedding can't represent meaning well with only 2–3 sentences.
- **Why not larger (300+ words)?** Risk of silent truncation by the model. Also, larger chunks make source citations less precise.
- **30-word overlap** prevents losing meaning at chunk boundaries — sentences that span two chunks are partially represented in both.

If switching to a model with a larger context window (e.g., `bge-large-en-v1.5` at 512 tokens), chunk size would increase to ~350 words.

### Hybrid search: vector + full-text via Reciprocal Rank Fusion

Pure vector search misses exact keyword matches (names, codes, acronyms). Pure BM25 misses synonyms and paraphrase. RRF merges both ranked lists without needing to normalise scores across different scales:

```
RRF score = 1/(60 + rank_vector) + 1/(60 + rank_fts)
```

The constant 60 is standard (from Cormack et al.). A chunk that ranks well in both searches scores higher than one that dominates only one. This runs entirely inside PostgreSQL — no third service needed.

### Metadata filtering

Every document stores a `doc_type`, `author`, `source`, and a free-form `extra_metadata` JSONB column. All search and chat endpoints accept these as optional filters, which translate to SQL `WHERE` clauses. This allows questions like "What do our Q1 memos say about the Nordic market?" without noise from unrelated documents.

**Important:** when two unrelated documents are in the knowledge base, always use metadata filters to restrict retrieval to the relevant scope. Without filters, the retrieval mixes context from all documents and the LLM may correctly say "I can't find that information" even when it exists.

---

## Stretch goals implemented

| Feature | Status | Notes |
|---|---|---|
| Hybrid search (vector + BM25) | ✅ Done | RRF over pgvector + PostgreSQL FTS |
| Streaming responses | ✅ Done | `/chat` with `"stream": true` returns SSE |
| Auth (API key) | ✅ Done | `X-Api-Key` header, enabled when `API_KEY` env var is set |
| Metadata filtering | ✅ Done | `doc_type`, `author`, `source`, `extra_metadata` on all endpoints |
| Re-ranking | ❌ Not done | See tradeoffs below |
| Graph representation | ❌ Not done | See tradeoffs below |

---

## What was left out and why

### Re-ranking
A cross-encoder (e.g., `ms-marco-MiniLM-L-6-v2`) re-scores the top-20 retrieved chunks before returning top-6. This would improve precision meaningfully. Skipped because it adds ~200 ms latency per query and another model download — a reasonable tradeoff for a foundation that already uses hybrid search.

### Graph representation
Entity extraction + a relations table would improve cross-document retrieval significantly. The approach: run NER (spaCy or a Claude call) on each chunk at index time, store `(entity, type, chunk_id)` triples, and at query time expand retrieval to include all chunks mentioning the same entities. Skipped due to time budget. The pgvector + FTS combination covers the primary retrieval use cases well without it.

### Contradiction handling
When two documents contradict each other, the current system retrieves both and passes both to Claude, which correctly hedges: *"Document A says X, but Document B says Y."* A production system would add a post-retrieval contradiction detection step — a second Claude call to compare high-scoring chunks with opposing claims and surface the conflict explicitly to the user.

### Model/API key selection in UI
The UI currently uses the API key from `.env`. A first-page setup screen where users enter their own API key and choose their LLM model would make this portable without touching configuration files. Deprioritised because the instructions explicitly say *"We're not evaluating UI/frontend."*

---

## How I used AI during this work

The primary tool was Claude Code (this same model) as a pair programmer. Here is an honest breakdown.

**What I delegated to the AI:**
- FastAPI router boilerplate and Pydantic schema definitions
- SQLAlchemy async session setup
- The RRF SQL query — I described the algorithm, the AI translated it to a CTE-based SQL query
- The streaming SSE implementation for the chat endpoint
- The frontend HTML/CSS/JS structure

**What I directed and corrected:**

| AI choice | My correction | Reason |
|---|---|---|
| `IVFFlat` vector index | Changed to `HNSW` | IVFFlat requires pre-defining cluster count and degrades with dynamic inserts |
| Character-based chunker | Changed to word-based | Character count maps poorly to the model's wordpiece token limit |
| Suggested running Postgres + Qdrant | Simplified to pgvector only | One service is simpler; SQL composability covers the filtering needs |
| `embedding <=> :param::vector` in SQLAlchemy | Changed to `CAST(:param AS vector)` | asyncpg couldn't parse the `::` cast next to a named parameter — caused a runtime syntax error |
| `anthropic.AsyncAnthropic()` with no key | Changed to pass `api_key=settings.anthropic_api_key` explicitly | Client initialised at import time before env vars were available |
| Granted DB-level privileges only | Added table-level `GRANT ALL ON ALL TABLES` | PostgreSQL distinguishes database access from table access — app got `permission denied` on first insert |

**Where I trusted the AI fully:**
- PyMuPDF PDF extraction (standard usage, no surprises)
- Pydantic v2 schema patterns
- The chunking sliding-window logic (correct first time, verified with a word-count test)

---

## Example requests

See [`examples.sh`](examples.sh) for a complete runnable curl script covering all endpoints.

```bash
# Upload plain text
curl -X POST http://localhost:8000/text \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Q1 Strategy Memo",
    "author": "Jane Smith",
    "doc_type": "memo",
    "source": "internal",
    "extra_metadata": {"year": "2024", "quarter": "Q1"},
    "text": "Our Q1 2024 focus is expanding into the Nordic market..."
  }'

# Upload a PDF
curl -X POST http://localhost:8000/document \
  -F "file=@report.pdf" \
  -F "title=Annual Report 2023" \
  -F "doc_type=report" \
  -F 'extra_metadata={"year":"2023"}'

# Search with metadata filter
curl "http://localhost:8000/search?q=Nordic+expansion&doc_type=memo"

# Chat (filtered to memos only)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our Q1 strategy?", "doc_type": "memo"}'

# Chat with streaming
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the strategy.", "stream": true}'

# List documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/{document_id}
```
