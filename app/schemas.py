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
