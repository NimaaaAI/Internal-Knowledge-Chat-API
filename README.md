# Internal Knowledge Chat API

A RAG (Retrieval-Augmented Generation) API: upload documents, search them by meaning, and chat with your knowledge base — with source references on every answer.

> This README is filled in progressively as the project is built.

---

## Step 1 — Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt. All packages installed after this go into the project's isolated environment, not system Python.

---

## Step 2 — Install and start PostgreSQL

PostgreSQL is a system-level service — it lives outside the venv.

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all
sudo service postgresql start
```

Verify it is running:

```bash
sudo service postgresql status
# Expected: 16/main (port 5432): online
```

### Unlock local access (dev only)

By default PostgreSQL uses `peer` auth on the local Unix socket, which requires you to be the `postgres` OS user. In dev, switch it to `trust` so any local user can connect without a password:

```bash
sudo bash -c "sed -i 's/local   all             postgres                                peer/local   all             postgres                                trust/' /etc/postgresql/16/main/pg_hba.conf"
sudo service postgresql restart
psql -U postgres
# You should see: postgres=#
```

---

## Step 3 — Create the project database and user

Inside the `postgres=#` shell, run each line one at a time:

```
CREATE USER rag WITH PASSWORD 'rag';
CREATE DATABASE knowledge OWNER rag;
GRANT ALL PRIVILEGES ON DATABASE knowledge TO rag;
\q
```

Expected output: `CREATE ROLE`, `CREATE DATABASE`, `GRANT`.

---

## Step 4 — Install pgvector

pgvector is a PostgreSQL **extension** — a compiled C plugin that adds vector storage and similarity search directly inside PostgreSQL. It is not a separate service or port.

```bash
cd /tmp
git clone --branch v0.8.2 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
cd /workspaces/Internal-Knowledge-Chat-API
```

What `sudo make install` does:
- Copies `vector.so` (the compiled library) to `/usr/lib/postgresql/16/lib/`
- Copies the SQL setup files to `/usr/share/postgresql/16/extension/`

pgvector is now **available** to PostgreSQL but not yet active in any database.

### Enable pgvector in the knowledge database

```bash
psql -U postgres -d knowledge
```

```
CREATE EXTENSION IF NOT EXISTS vector;
```

This tells PostgreSQL to load `vector.so` into the `knowledge` database and register the `vector` data type and `<=>` distance operator.

### How everything connects

There is only one process and one port (5432). pgvector lives inside PostgreSQL — like a browser extension lives inside the browser.

```
Python app
    │
    │  postgresql://rag:rag@localhost:5432/knowledge
    ▼
PostgreSQL on port 5432
    └── knowledge database
            ├── vector extension  ← loaded from vector.so, handles embeddings
            ├── documents table   ← stores metadata
            └── chunks table      ← stores text + vector column
```

### Verify pgvector works end-to-end

Inside `psql -U postgres -d knowledge`, run:

```sql
CREATE TABLE test_vectors (id serial, embedding vector(3));
INSERT INTO test_vectors (embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[1,2,4]');
SELECT id, embedding, embedding <=> '[1,2,3]'::vector AS distance
FROM test_vectors ORDER BY distance;
DROP TABLE test_vectors;
```

Expected: row 1 returns distance `0`, row 3 next, row 2 last.
This confirms vector storage, similarity search, and distance ordering all work.

---

## Step 5 — Create the project folder structure

```bash
mkdir -p app/routes app/static
touch app/__init__.py app/routes/__init__.py
```

- `app/routes/` — one file per endpoint group (upload, search, chat, documents)
- `app/static/` — the frontend HTML file
- `__init__.py` files tell Python to treat these folders as packages

---

## Step 6 — Environment variables

```bash
cp .env.example .env
# Open .env and fill in your ANTHROPIC_API_KEY
```

Never commit `.env` — it is listed in `.gitignore`. The `.env.example` file shows the required keys without real values.

---

## Step 7 — Install Python dependencies

Dependencies are added to `requirements.txt` as they are needed. Install everything at once with:

```bash
pip install -r requirements.txt
```

Current dependencies and why each is needed:

| Package | Purpose |
|---|---|
| pydantic-settings | reads `.env` into typed Python settings |
| sqlalchemy[asyncio] | async ORM for PostgreSQL |
| asyncpg | async PostgreSQL driver used by SQLAlchemy |
| pgvector | SQLAlchemy type for the `vector` column |

---

## Step 8 — Apply the database schema

```bash
psql -U postgres -d knowledge -f init.sql
```

Creates two tables:

**`documents`** — one row per uploaded file or text block

| Column | Type | Purpose |
|---|---|---|
| id | UUID | primary key |
| title | TEXT | human-readable name |
| source | TEXT | where it came from |
| author | TEXT | who wrote it |
| doc_type | TEXT | memo, report, article, etc. |
| content_type | TEXT | `text` or `document` (PDF) |
| extra_metadata | JSONB | any extra key-value pairs for filtering |
| created_at | TIMESTAMPTZ | upload timestamp |

**`chunks`** — one row per piece of text split from a document

| Column | Type | Purpose |
|---|---|---|
| id | UUID | primary key |
| document_id | UUID | foreign key → documents.id |
| chunk_index | INTEGER | position within the document |
| text | TEXT | the raw chunk text |
| embedding | vector(384) | 384-dimension embedding vector |
| ts_vector | TSVECTOR | auto-updated full-text search index |
| created_at | TIMESTAMPTZ | creation timestamp |

Indexes: HNSW (vector search), GIN (full-text search), B-tree (document lookup), GIN (JSONB metadata filtering).

---

## Step 9 — Core Python modules

### `app/config.py`
Reads all environment variables into a typed `Settings` object. Every other module imports from here — nothing is hardcoded.

### `app/database.py`
Creates the async SQLAlchemy engine and a `get_db` dependency that FastAPI injects into route handlers.

### `app/models.py`
SQLAlchemy ORM classes that map to the `documents` and `chunks` tables. Uses `pgvector.sqlalchemy.Vector` for the embedding column.

### `app/schemas.py`
Pydantic models that define the shape of every API request and response. Pydantic validates incoming data automatically.

### `app/chunking.py`
Splits a document into overlapping word-based chunks. 150 words per chunk, 30-word overlap. Word-based (not character-based) because it maps predictably to the embedding model's 256 wordpiece token limit.

---

## Step 5 — Create the project folder structure

```bash
mkdir -p app/routes app/static
touch app/__init__.py app/routes/__init__.py
```

- `app/routes/` — one file per endpoint group (upload, search, chat, documents)
- `app/static/` — the frontend HTML file
- `__init__.py` files tell Python to treat these folders as packages

---

## Step 6 — Apply the database schema

```bash
psql -U postgres -d knowledge -f init.sql
```

This creates two tables:

**`documents`** — one row per uploaded file or text block
| Column | Type | Purpose |
|---|---|---|
| id | UUID | primary key |
| title | TEXT | human-readable name |
| source | TEXT | where it came from (URL, filename, etc.) |
| author | TEXT | who wrote it |
| doc_type | TEXT | memo, report, article, etc. |
| content_type | TEXT | 'text' or 'document' (PDF) |
| extra_metadata | JSONB | any extra key-value pairs for filtering |
| created_at | TIMESTAMPTZ | when it was uploaded |

**`chunks`** — one row per chunk of text split from a document
| Column | Type | Purpose |
|---|---|---|
| id | UUID | primary key |
| document_id | UUID | foreign key → documents.id |
| chunk_index | INTEGER | position within the document |
| text | TEXT | the raw chunk text |
| embedding | vector(384) | the 384-dimension embedding vector |
| ts_vector | TSVECTOR | auto-updated full-text search index |
| created_at | TIMESTAMPTZ | when it was created |

**Indexes created:**
- `HNSW` on `chunks.embedding` — fast approximate nearest-neighbour vector search
- `GIN` on `chunks.ts_vector` — fast full-text keyword search
- B-tree on `chunks.document_id` — fast lookup of all chunks for a document
- `GIN` on `documents.extra_metadata` — fast JSONB key-value filtering

**Trigger:** `sync_chunk_ts_vector` — automatically populates `ts_vector` from `text` on every insert or update, so full-text search is always up to date.

---


