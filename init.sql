CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title         TEXT        NOT NULL,
    source        TEXT,
    author        TEXT,
    doc_type      TEXT,
    content_type  TEXT        NOT NULL DEFAULT 'text',
    extra_metadata JSONB      NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER     NOT NULL,
    text         TEXT        NOT NULL,
    embedding    vector(384),
    ts_vector    TSVECTOR,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Automatically update ts_vector whenever a chunk's text is inserted or changed
CREATE OR REPLACE FUNCTION sync_chunk_ts_vector()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.ts_vector := to_tsvector('english', NEW.text);
    RETURN NEW;
END;
$$;

CREATE TRIGGER chunk_ts_vector_trigger
    BEFORE INSERT OR UPDATE OF text ON chunks
    FOR EACH ROW EXECUTE FUNCTION sync_chunk_ts_vector();

-- HNSW index: fast approximate nearest-neighbour search on the embedding column
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- GIN index: fast full-text search on ts_vector
CREATE INDEX IF NOT EXISTS chunks_ts_vector_idx
    ON chunks USING GIN (ts_vector);

-- Regular index for looking up all chunks belonging to a document
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

-- GIN index on extra_metadata for fast key-value filtering
CREATE INDEX IF NOT EXISTS documents_metadata_idx
    ON documents USING GIN (extra_metadata);

-- Grant the app user full access to all tables and sequences
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag;
