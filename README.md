# Internal Knowledge Chat API

---

## Part 1 — What this project is

### The core idea: RAG

This is a **RAG** system — Retrieval-Augmented Generation.

The problem it solves: an LLM like Claude knows a lot about the world, but it knows nothing about *your* internal documents. If you ask it "What did our Q1 memo say about the Nordic market?", it cannot answer — it has never seen that memo.

RAG fixes this in two steps:
1. **Retrieve** — search your document library and find the most relevant passages
2. **Generate** — pass those passages to the LLM as context, then ask it your question

The LLM now has the right information in front of it and can give a grounded answer. Every answer includes the source documents it used, so you can verify the claim.

---

### How the search works (step by step)

When you ask a question, the API runs this pipeline:

#### Step 1 — Embed the query
Your question is converted to a list of 384 numbers (a "vector") using `all-MiniLM-L6-v2`. This captures the *meaning* of the question, not just the keywords.

#### Step 2 — Hybrid search (VEC + FTS)
Two searches run at the same time:

- **VEC (Vector Search)** — finds chunks whose meaning is mathematically close to your question. Good for paraphrasing, synonyms, and conceptual similarity. Runs inside PostgreSQL using the `pgvector` extension with an HNSW index.
- **FTS (Full-Text Search)** — finds chunks that contain the exact keywords from your question. Good for names, codes, and acronyms. Runs inside PostgreSQL using built-in `tsvector` / `GIN` index.

The results of both searches are then merged using **RRF (Reciprocal Rank Fusion)**:

```
RRF score = 1/(60 + rank_vector) + 1/(60 + rank_fts)
```

A chunk that ranks well in *both* searches scores higher than one that only dominates one. The constant 60 is a standard dampening factor that prevents the single top result from dominating everything.

#### Step 3 — Graph expansion (optional)
At upload time, Claude reads each chunk and extracts named entities (people, organisations, places). These are stored in an `entities` table linked to their chunk.

At query time, the top entities found across the retrieved chunks are used to pull in *extra* chunks from the database that mention the same entities but weren't in the original search results. This increases recall — you find related information you would have missed.

#### Step 4 — Re-ranking
The expanded pool of chunks is re-scored by a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`). Unlike the embedding model (which reads the query and chunk *separately*), the cross-encoder reads them *together* — it can understand how well this specific passage answers this specific question. Top 6 are kept.

#### Step 5 — Answer generation
The final chunks are assembled into a context block and sent to Claude along with your question. Claude generates an answer and cites which documents it drew from.

---

### Models used

| Model | Purpose | Why this model |
|---|---|---|
| `all-MiniLM-L6-v2` | Text → 384-dim vector (embedding) | Fast (~80 ms/chunk on CPU), free, no API key, runs locally |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-rank retrieved chunks by relevance | More accurate than embedding similarity — reads query+passage together |
| `claude-sonnet-4-6` (Anthropic) | Generate answers + extract entities | Strong instruction-following, honest about uncertainty, good at structured output |

---

### Features

| Feature | Description |
|---|---|
| Upload plain text | POST with JSON — text is chunked and embedded server-side |
| Upload PDF or .txt | File upload — text extracted, chunked, embedded |
| Metadata | Every document has `title`, `author`, `source`, `doc_type`, and free-form `extra_metadata`. All search and chat endpoints accept these as filters. |
| Hybrid search | Vector similarity + full-text search merged by RRF |
| Re-ranking | Cross-encoder scores top candidates before returning results |
| Graph expansion | Entity-aware retrieval pulls related chunks by entity co-occurrence |
| Streaming chat | Chat answers stream token-by-token over SSE |
| Entity browser | Search all extracted entities by name or type (PERSON, ORGANIZATION, PLACE) |
| Co-occurrence search | Find chunks that mention multiple entities together |
| Retrieval trace | Inspect how any query scores through the full pipeline — vector rank, FTS rank, RRF score, cross-encoder score |
| Pipeline Inspector | Watch each step of the RAG pipeline run live in your browser with timing |
| API key auth | `X-Api-Key` header required when `API_KEY` env var is set |
| Web UI | Built-in interface at `http://localhost:8000` with tabs for Chat, Search, Documents, and Internals |
| OpenAPI docs | Auto-generated docs at `http://localhost:8000/docs` |

---

### Architecture

Everything runs inside a single PostgreSQL database — no separate vector database needed.

```
documents          — one row per uploaded document + metadata
chunks             — text fragments of ~150 words each
  └─ embedding     — 384-dim vector (HNSW index for vector search)
  └─ ts_vector     — full-text index (GIN index for keyword search)
entities           — extracted named entities linked to their chunk
```

The API is built with **FastAPI** (Python), using async throughout. Embeddings run on CPU. The re-ranker also runs on CPU. The only external API call is to Anthropic (for generating answers and extracting entities).

---

## Part 2 — How to set up and run

### What you need

- Python 3.12+
- Git
- An Anthropic API key (from [console.anthropic.com](https://console.anthropic.com))

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/NimaaaAI/Internal-Knowledge-Chat-API.git
cd Internal-Knowledge-Chat-API
python3 -m venv .venv
source .venv/bin/activate
```

---

### Step 2 — Install PostgreSQL

PostgreSQL is the database that stores everything — documents, chunks, vectors, and entities. It is a system service, not a Python package.

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
sudo service postgresql start
```

> **Every time you restart your machine or Codespace**, PostgreSQL stops. Run `sudo service postgresql start` again before starting the API.

---

### Step 3 — Allow local connections (development only)

By default PostgreSQL requires you to be the `postgres` OS user to connect. Switch it to trust-mode for local development:

```bash
sudo bash -c "sed -i 's/local   all             postgres                                peer/local   all             postgres                                trust/' /etc/postgresql/16/main/pg_hba.conf"
sudo service postgresql restart
```

---

### Step 4 — Create the database and user

```bash
psql -U postgres
```

Inside the psql shell, run:

```sql
CREATE USER rag WITH PASSWORD 'rag';
CREATE DATABASE knowledge OWNER rag;
GRANT ALL PRIVILEGES ON DATABASE knowledge TO rag;
\q
```

---

### Step 5 — Install pgvector

pgvector is a PostgreSQL extension that adds vector storage and similarity search. It is not a separate service — it runs inside your existing PostgreSQL.

```bash
cd /tmp
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
cd -
```

---

### Step 6 — Create the database schema

This creates all tables, indexes, and grants the right permissions:

```bash
psql -U postgres -d knowledge -f init.sql
```

---

### Step 7 — Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

The other values (database URL, API key for auth) have working defaults for local development.

---

### Step 8 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This downloads the embedding model and cross-encoder model on first run (about 300 MB total). They are cached locally after that.

---

### Step 9 — Start the API

```bash
uvicorn app.main:app --reload
```

- Web UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

---

### Troubleshooting

#### "Connection refused" when using the API

PostgreSQL is not running. Fix:

```bash
sudo service postgresql start
```

This happens every time you restart your machine or Codespace. Make it a habit to run this before starting the API.

#### "Permission denied" on first insert

The database user `rag` does not have table-level access. PostgreSQL distinguishes database access from table access. Fix: re-run the schema file, which includes the correct GRANT statements:

```bash
psql -U postgres -d knowledge -f init.sql
```

#### "relation entities does not exist"

The `entities` table was not created. This happens if you set up the database with an older version of `init.sql` before the entity graph feature was added. Fix: run this once:

```bash
psql -U postgres -d knowledge -c "
CREATE TABLE IF NOT EXISTS entities (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id      UUID NOT NULL REFERENCES chunks(id)     ON DELETE CASCADE,
    document_id   UUID NOT NULL REFERENCES documents(id)  ON DELETE CASCADE,
    name          TEXT NOT NULL,
    type          TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities (lower(name));
CREATE INDEX IF NOT EXISTS idx_entities_type  ON entities (type);
CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities (chunk_id);
CREATE INDEX IF NOT EXISTS idx_entities_doc   ON entities (document_id);
GRANT ALL ON entities TO rag;
"
```

#### Models downloading slowly on first start

The first startup downloads `all-MiniLM-L6-v2` (~90 MB) and `ms-marco-MiniLM-L-6-v2` (~80 MB) from HuggingFace. This is normal — it only happens once. After that, they are cached in `~/.cache/huggingface/`.

#### The app starts but entity extraction is slow

Entity extraction calls the Anthropic API once per chunk — if you upload a large document with many chunks, this takes time. The API runs up to 5 extractions in parallel (controlled by `GRAPH_CONCURRENCY` in settings). This is by design to avoid hitting Anthropic rate limits.

---

### API endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/text` | Upload plain text with metadata |
| POST | `/document` | Upload a PDF or .txt file |
| GET | `/search` | Hybrid search with optional re-ranking |
| POST | `/chat` | Ask a question, get an answer with source citations |
| GET | `/documents` | List all uploaded documents |
| DELETE | `/documents/{id}` | Delete a document and all its data |
| GET | `/debug/stats` | Document and chunk counts |
| GET | `/debug/chunks` | Browse all chunks for a document |
| GET | `/debug/trace` | Full retrieval trace for a query |
| GET | `/debug/entities` | Search extracted entities |
| GET | `/debug/cooccurrence` | Find chunks mentioning multiple entities |
| GET | `/debug/pipeline` | Live pipeline inspection (streaming) |

---

### Example requests

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

# Search with metadata filter
curl "http://localhost:8000/search?q=Nordic+expansion&doc_type=memo"

# Chat (streaming)
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our Q1 strategy?", "doc_type": "memo", "stream": true}'

# List documents
curl http://localhost:8000/documents

# Delete a document
curl -X DELETE http://localhost:8000/documents/{document_id}
```

See [`examples.sh`](examples.sh) for a full runnable script covering all endpoints.
