# Internal Knowledge Chat API

A production-ready RAG (Retrieval-Augmented Generation) system that lets teams upload internal documents and query them in natural language. Every answer is grounded in the source material and includes precise citations so claims can be verified.

Built with Python, FastAPI, PostgreSQL, and Claude.

---

## Table of Contents

1. [How to Run — Docker (Recommended)](#1-how-to-run--docker-recommended)
2. [How to Run — Manual Setup](#2-how-to-run--manual-setup)
3. [API Reference](#3-api-reference)
4. [Architecture and Design Decisions](#4-architecture-and-design-decisions)
5. [What Was Implemented](#5-what-was-implemented)
6. [What Was Left Out and Why](#6-what-was-left-out-and-why)
7. [How I Used AI During This Work](#7-how-i-used-ai-during-this-work)

---

## 1. How to Run — Docker (Recommended)

The entire stack (API + PostgreSQL + pgvector) runs in two containers. No database installation, no Python environment setup required.

**Requirements:** Docker, Docker Compose, an Anthropic API key.

```bash
git clone https://github.com/NimaaaAI/Internal-Knowledge-Chat-API.git
cd Internal-Knowledge-Chat-API
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env
docker compose up
```

Open **http://localhost:8000** in your browser.

> **First run:** the API downloads the embedding and re-ranking models (~300 MB) on startup. This takes about one minute and only happens once — models are cached in a Docker volume and reused on every subsequent start.

To stop:
```bash
docker compose down
```

All uploaded documents and data persist between restarts. To wipe everything and start fresh:
```bash
docker compose down -v
```

---

## 2. How to Run — Manual Setup

Use this for local development with hot-reload.

### Prerequisites
- Python 3.12+
- PostgreSQL 16
- An Anthropic API key

### Step 1 — Clone and create the virtual environment

```bash
git clone https://github.com/NimaaaAI/Internal-Knowledge-Chat-API.git
cd Internal-Knowledge-Chat-API
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2 — Install PostgreSQL

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
sudo service postgresql start
```

> PostgreSQL stops when your machine restarts. Run `sudo service postgresql start` before starting the API each time.

### Step 3 — Allow local connections

```bash
sudo bash -c "sed -i 's/local   all             postgres                                peer/local   all             postgres                                trust/' /etc/postgresql/16/main/pg_hba.conf"
sudo service postgresql restart
```

### Step 4 — Create the database

```bash
psql -U postgres -c "CREATE DATABASE knowledge;"
```

### Step 5 — Install pgvector

```bash
cd /tmp
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
cd -
```

### Step 6 — Apply the schema

```bash
psql -U postgres -d knowledge -f init.sql
```

### Step 7 — Configure environment

```bash
cp .env.example .env
# Open .env and fill in your ANTHROPIC_API_KEY
```

### Step 8 — Install dependencies and start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- Web UI: `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Connection refused` | PostgreSQL not running | `sudo service postgresql start` |
| `Permission denied` on insert | Table-level grants missing | Re-run `psql -U postgres -d knowledge -f init.sql` |
| `relation "entities" does not exist` | Schema applied before entities table was added | Re-run `init.sql` |
| Slow first startup | Models downloading (~300 MB) | Wait — only happens once, then cached |

---

## 3. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/text` | Upload plain text with metadata |
| `POST` | `/document` | Upload a PDF or .txt file |
| `GET` | `/search` | Hybrid search with optional re-ranking |
| `POST` | `/chat` | Ask a question, get an answer with source citations |
| `GET` | `/documents` | List all uploaded documents |
| `DELETE` | `/documents/{id}` | Delete a document and all its data |
| `GET` | `/debug/stats` | Document and chunk counts |
| `GET` | `/debug/trace` | Full retrieval trace with all ranking scores |
| `GET` | `/debug/entities` | Search extracted entities by name or type |
| `GET` | `/debug/cooccurrence` | Find chunks mentioning multiple entities together |
| `GET` | `/debug/pipeline` | Live step-by-step pipeline inspection (SSE) |

See [`examples.sh`](examples.sh) for runnable curl examples covering all endpoints.

### Example: upload and query

```bash
# Upload a document
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

# Ask a question with streaming
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our Q1 strategy?", "doc_type": "memo", "stream": true}'
```

---

## 4. Architecture and Design Decisions

### Stack

| Layer | Choice | Rationale |
|---|---|---|
| Framework | FastAPI (Python) | Async-native, auto OpenAPI docs, fast to iterate |
| Database | PostgreSQL 16 + pgvector | One service handles vectors, full-text search, relational joins, and metadata filtering — no separate vector DB needed |
| Embedding model | `all-MiniLM-L6-v2` (384-dim) | Free, no API key, runs on CPU in ~80 ms/chunk, no external dependency |
| Re-ranking model | `ms-marco-MiniLM-L-6-v2` | Cross-encoder trained on MS MARCO — reads query and passage together for precise relevance scoring |
| LLM | Claude Sonnet (Anthropic) | Strong instruction-following, reliable structured JSON output for entity extraction |
| PDF extraction | PyMuPDF | Fast, layout-aware, ships as a compiled wheel with no system dependencies |

### Why PostgreSQL instead of a dedicated vector database?

A dedicated vector database (Qdrant, Weaviate) requires running a second service alongside PostgreSQL for relational data. With pgvector, one database handles everything — vector similarity search, full-text search, relational joins, and metadata filtering — all composable in a single SQL query. At knowledge-base scale (under one million chunks), pgvector with HNSW matches the retrieval quality of dedicated solutions while eliminating operational complexity.

If the chunks grew to tens of millions of chunks with sub-10 ms latency requirements, Qdrant would be the right call.

### Chunking strategy

`all-MiniLM-L6-v2` has a hard limit of 256 wordpiece tokens. English text averages ~1.4 wordpieces per word, so a 150-word ceiling keeps every chunk safely within the model's window with headroom for subword-heavy content.

The chunker is paragraph-aware and recursive rather than a fixed sliding window:

1. **Split on paragraph boundaries** (`\n\n`) first. A paragraph that already fits within 150 words becomes one chunk — its natural boundaries are preserved.
2. **Fall back to sentences** for any paragraph that exceeds 150 words. The paragraph is split on sentence-ending punctuation and sentences are accumulated until the limit is reached.
3. **Overlap by carrying the last unit.** When a new chunk starts, the final paragraph or sentence from the previous chunk is included as the opening unit. This ensures that a thought split across a boundary is fully represented in at least one chunk.

This produces semantically coherent chunks instead of cuts that land mid-sentence, which improves both embedding quality and the relevance of source citations returned to the user.

**What was considered and not used:**

- *LLM-based chunking* — Claude could read each document and decide where the natural topic boundaries are, producing semantically ideal chunks. The tradeoff is cost and latency: every upload would require one or more LLM calls just for chunking, before embedding and entity extraction. At the scale of a knowledge base this adds up quickly. The paragraph-aware strategy captures most of the benefit at zero API cost.
- *Larger embedding model* — `bge-large-en-v1.5` (1024-dim, 512-token window) would improve retrieval quality for domain-specific language and allow chunks up to ~350 words. The tradeoff is memory and CPU time: the model is roughly 5× larger than `all-MiniLM-L6-v2`. The cross-encoder re-ranker compensates for much of the quality gap in practice, so the smaller model was kept as the pragmatic baseline.

### Retrieval pipeline

The full pipeline for every query:

**Step 1 — Hybrid search (VEC + FTS → RRF)**

Two searches run in parallel inside a single SQL query:

- **Vector search** — cosine distance between the query embedding and all stored chunk embeddings via HNSW index. Captures semantic similarity, synonyms, and paraphrase.
- **Full-text search** — PostgreSQL `ts_rank` over `tsvector` via GIN index. Captures exact keyword matches, names, and codes that vector search can miss.

Results are merged with Reciprocal Rank Fusion:

```
RRF score = 1/(60 + rank_vector) + 1/(60 + rank_fts)
```

**Step 2 — Graph expansion**

At upload time, Claude reads each chunk and extracts named entities (people, organisations, places), storing them in an `entities` table linked to their chunk. At query time, the most-mentioned entities across the retrieved chunks are used to pull in additional chunks that share those entities but did not rank highly in the vector or full-text search. This increases recall for cross-document queries without adding a separate graph database.


**Step 3 — Cross-encoder re-ranking**

The expanded pool of candidates is re-scored by the cross-encoder, which reads the query and each passage together in the same attention layers. The cross-encoder directly models how well a specific passage answers a specific question. The top 6 candidates are kept for the LLM.

**Step 4 — Answer generation**

The final chunks are formatted into a labelled context block and sent to Claude with a system prompt that restricts answers to the provided context and requires inline source citations in the format `[Doc: title, chunk N]`. Streaming responses are delivered token by token via Server-Sent Events.


**Pipeline Inspector**

Every step of the pipeline can be observed live in the browser, including timing, which entities were used for graph expansion, how chunks moved in ranking after re-ranking, and the exact context sent to Claude.

### Data model

```
documents
  id               UUID primary key
  title            TEXT
  author           TEXT        filterable
  source           TEXT        filterable
  doc_type         TEXT        filterable (memo / report / article / ...)
  extra_metadata   JSONB       free-form key-value pairs, GIN-indexed
  content_type     TEXT        'text' | 'document'
  created_at       TIMESTAMPTZ

chunks
  id               UUID primary key
  document_id      UUID → documents  (CASCADE DELETE)
  chunk_index      INTEGER
  text             TEXT
  embedding        vector(384)  HNSW index — vector search
  ts_vector        TSVECTOR     GIN index  — full-text search
  created_at       TIMESTAMPTZ

entities
  id               UUID primary key
  chunk_id         UUID → chunks     (CASCADE DELETE)
  document_id      UUID → documents  (CASCADE DELETE)
  name             TEXT
  type             TEXT        PERSON | ORGANIZATION | PLACE
  created_at       TIMESTAMPTZ
```

Deleting a document cascades automatically through its chunks and entities.


### Upload flow

When a document is uploaded, embedding (CPU) and entity extraction (Claude API) run concurrently using `asyncio.gather`. This means the network round-trips to Claude overlap with local CPU work — reducing upload time compared to running them sequentially.


---

## 5. What Was Implemented

All five must-have requirements and all five stretch goals were completed.

| Feature | Status |
|---|---|
| `POST /text` — plain text upload with metadata | ✅ |
| `POST /document` — PDF and .txt upload, text extracted server-side | ✅ |
| Chunking, embeddings, persistent storage | ✅ |
| `GET /search` — chunks with score and document reference | ✅ |
| `POST /chat` — retrieval + LLM answer + source citations | ✅ |
| Persistence across restarts | ✅ |
| Hybrid search (vector + BM25/FTS) with RRF | ✅ |
| Streaming responses via SSE | ✅ |
| Cross-encoder re-ranking | ✅ |
| API key authentication | ✅ |
| Graph representation — entity extraction and retrieval expansion | ✅ |
| Metadata filtering on all search and chat endpoints | ✅ |
| Docker Compose — zero-configuration deployment | ✅ |

---

## 6. What Was Left Out and Why

### Contradiction detection

When two documents make conflicting claims, the current system retrieves both and passes them to Claude, which correctly hedges: *"Document A says X, but Document B says Y."* A production system would add a dedicated post-retrieval step — a second Claude call to compare high-scoring chunks with opposing claims and surface the conflict explicitly to the user. This was deprioritised in favour of completing the graph expansion and re-ranking features, which have broader impact on the common case.

### Incremental document updates

There is no `PATCH /documents/{id}` endpoint. Updating a document requires deleting it and re-uploading. For a production system, partial re-indexing would reduce latency and API cost. Deprioritised because it adds significant complexity for a feature that matters less at typical knowledge-base scale.

### Larger embedding model

`bge-large-en-v1.5` (1024-dim) or OpenAI's `text-embedding-3-small` would improve retrieval quality, particularly for domain-specific language. The tradeoff is memory, CPU time, and an additional API key. `all-MiniLM-L6-v2` was chosen as the pragmatic baseline — the cross-encoder re-ranker compensates for much of the quality gap in practice.

### Production instrumentation

Structured logging, metrics, CI/CD pipelines, and rate limiting were not implemented. The brief explicitly excluded these from evaluation.

---

## 7. How I Used AI During This Work

The primary tool was Claude Code as a pair programmer throughout the project. The key principle was to guide the AI step by step — not hand over the whole problem.

**Development approach**

The project was built incrementally: virtual environment first, then dependencies one by one, then each feature individually — upload, search, chat, streaming, re-ranking, graph expansion, Docker. Each step was tested and verified before moving to the next. This meant bugs were caught in isolation, the code was understood at each stage, and the architecture evolved based on what was actually needed rather than what was anticipated upfront.

**What I delegated to the AI:**
- FastAPI router boilerplate and Pydantic schema definitions
- The RRF SQL query — I described the algorithm, the AI translated it into a CTE-based query
- The SSE streaming implementation for the chat endpoint
- The Pipeline Inspector frontend — event rendering and step state transitions
- HTML structure and layout

**What I directed:**

Every significant architectural decision was made before any code was written. The choice of PostgreSQL over a dedicated vector database, the chunking strategy based on the model's token limit, and the three-stage retrieval pipeline (hybrid → graph → re-rank) were all defined upfront. The AI implemented what I specified, not the other way around.

For Docker, rather than letting the AI generate a complete setup in one shot, I guided it to understand each component first — what the pgvector image does, why the healthcheck matters, how volumes work — before writing the files. This meant I could verify the setup was correct rather than just hoping it worked.

**Where I corrected the AI:**

| AI choice | My correction | Reason |
|---|---|---|
| Character-based chunker | Changed to word-based | Character count maps poorly to the model's wordpiece token limit |
| Suggested Postgres + Qdrant | Simplified to pgvector only | One service is simpler; SQL composability covers all filtering needs |
| `AsyncAnthropic()` with no key | Changed to pass `api_key` explicitly | Client initialised at import time before environment variables were loaded |
| Database-level grants only | Added table-level `GRANT ALL ON ALL TABLES` | PostgreSQL distinguishes database access from table access |
| Sequential embed then extract | Restructured to `asyncio.gather` | Embedding (CPU) and entity extraction (network) are independent — running them concurrently reduced upload time |

**Where I trusted the AI fully:**
- PyMuPDF PDF extraction (standard usage, no surprises)
- Pydantic v2 schema patterns
- The sliding-window chunking logic (correct on first attempt, verified with a manual word-count test)
- Docker Compose syntax for the pgvector image

