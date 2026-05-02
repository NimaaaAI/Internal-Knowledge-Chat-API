import asyncio
import json
from typing import Any

import fitz
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import verify_api_key
from app.database import get_db
from app.entity_extraction import extract_all_entities
from app.models import Document, Chunk, Entity
from app.schemas import TextUploadRequest, UploadResponse
from app.chunking import chunk_text
from app.embeddings import embed
from app.extraction import extract_text_from_pdf

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/upload/preview")
async def preview_upload(file: UploadFile = File(...)):
    """
    Read a file's built-in metadata without saving anything.
    Called by the UI when a file is selected so the form can be pre-filled.
    """
    raw_bytes = await file.read()
    filename = file.filename or "untitled"
    content_type = file.content_type or ""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename

    result: dict[str, Any] = {
        "title": stem,
        "author": None,
        "page_count": None,
        "file_size_bytes": len(raw_bytes),
        "word_count": None,
    }

    if "pdf" in content_type or filename.lower().endswith(".pdf"):
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            meta = doc.metadata
            result["title"] = (meta.get("title") or "").strip() or stem
            result["author"] = (meta.get("author") or "").strip() or None
            result["page_count"] = doc.page_count
            doc.close()
        except Exception:
            pass
    elif "text" in content_type or filename.lower().endswith(".txt"):
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
            result["word_count"] = len(text.split())
        except Exception:
            pass

    return result


async def _index_document(
    db: AsyncSession,
    title: str,
    text: str,
    source: str | None,
    author: str | None,
    doc_type: str | None,
    content_type: str,
    extra_metadata: dict[str, Any],
) -> UploadResponse:
    """Shared logic: chunk → embed → save to DB."""
    # Auto-computed stats — user-provided keys in extra_metadata take priority
    auto_stats: dict[str, Any] = {"word_count": len(text.split())}
    extra_metadata = {**auto_stats, **extra_metadata}

    doc = Document(
        title=title,
        source=source,
        author=author,
        doc_type=doc_type,
        content_type=content_type,
        extra_metadata=extra_metadata,
    )
    db.add(doc)
    await db.flush()  # get doc.id before committing

    raw_chunks = chunk_text(text)
    if not raw_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document produced no chunks — is the text empty?",
        )

    # Embed (CPU) and extract entities (Claude API) run concurrently
    vectors, entity_results = await asyncio.gather(
        asyncio.to_thread(embed, raw_chunks),
        extract_all_entities(raw_chunks),
    )

    # Save chunks first so we have their IDs for entity records
    chunk_objs: list[Chunk] = []
    for idx, (chunk_text_, vector) in enumerate(zip(raw_chunks, vectors)):
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=idx,
            text=chunk_text_,
            embedding=vector,
        )
        db.add(chunk)
        chunk_objs.append(chunk)

    await db.flush()  # assigns chunk IDs without committing

    # Save entities linked to their chunk
    for chunk, entities in zip(chunk_objs, entity_results):
        for ent in entities:
            db.add(Entity(
                chunk_id=chunk.id,
                document_id=doc.id,
                name=ent["name"],
                type=ent["type"],
            ))

    await db.commit()
    return UploadResponse(document_id=doc.id, title=doc.title, chunks_created=len(raw_chunks))


@router.post("/text", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_text(payload: TextUploadRequest, db: AsyncSession = Depends(get_db)):
    """Upload plain text with metadata. Chunked and embedded server-side."""
    return await _index_document(
        db=db,
        title=payload.title,
        text=payload.text,
        source=payload.source,
        author=payload.author,
        doc_type=payload.doc_type,
        content_type="text",
        extra_metadata=payload.extra_metadata,
    )


@router.post("/document", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    source: str | None = Form(default=None),
    author: str | None = Form(default=None),
    doc_type: str | None = Form(default=None),
    extra_metadata: str = Form(default="{}"),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF file. Text is extracted server-side, then chunked and embedded."""
    raw_bytes = await file.read()

    filename = (file.filename or "").lower()
    content_type = file.content_type or ""
    file_stats: dict[str, Any] = {"file_size_bytes": len(raw_bytes)}

    if "pdf" in content_type or filename.endswith(".pdf"):
        try:
            text = await asyncio.to_thread(extract_text_from_pdf, raw_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"PDF extraction failed: {exc}",
            )
        # Read page count and author fallback from PDF metadata
        try:
            _pdf = fitz.open(stream=raw_bytes, filetype="pdf")
            file_stats["page_count"] = _pdf.page_count
            if not author:
                author = (_pdf.metadata.get("author") or "").strip() or None
            _pdf.close()
        except Exception:
            pass
    elif "text" in content_type or filename.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="replace")
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and plain-text files are supported",
        )

    try:
        meta = json.loads(extra_metadata)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="extra_metadata must be valid JSON",
        )

    # file_stats are the base; user-provided meta keys win
    meta = {**file_stats, **meta}

    return await _index_document(
        db=db,
        title=title,
        text=text,
        source=source,
        author=author,
        doc_type=doc_type,
        content_type="document",
        extra_metadata=meta,
    )
