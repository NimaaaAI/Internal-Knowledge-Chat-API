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


