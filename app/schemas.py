import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class TextUploadRequest(BaseModel):
    text: str = Field(..., min_length=1)
    title: str
    source: str | None = None
    author: str | None = None
    doc_type: str | None = None          # e.g. 'memo', 'report', 'article'
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    title: str
    chunks_created: int


# ── Documents list ────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    source: str | None
    author: str | None
    doc_type: str | None
    content_type: str
    extra_metadata: dict[str, Any]
    created_at: datetime
    chunk_count: int


# ── Search ────────────────────────────────────────────────────────────────────

class ChunkResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    text: str
    score: float
    document_title: str
    document_source: str | None
    document_author: str | None
    doc_type: str | None
    extra_metadata: dict[str, Any]
    vector_rank: int | None = None   # rank in vector search (None = not found by vector)
    fts_rank: int | None = None      # rank in full-text search (None = not found by FTS)


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_type: str | None = None
    author: str | None = None
    source: str | None = None
    extra_metadata_filter: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False


class SourceReference(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]


# ── Debug / Internals ─────────────────────────────────────────────────────────

class DebugStats(BaseModel):
    document_count: int
    chunk_count: int


class ChunkDetail(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    text: str
    embedding_dims: int


class TraceResult(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    chunk_index: int
    text: str
    vector_rank: int | None
    fts_rank: int | None
    rrf_score: float
    rerank_score: float | None = None   # populated only when rerank=true in trace


class TraceResponse(BaseModel):
    query: str
    results: list[TraceResult]
