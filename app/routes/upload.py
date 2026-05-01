import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import verify_api_key
from app.database import get_db
from app.models import Document, Chunk
from app.schemas import TextUploadRequest, UploadResponse
from app.chunking import chunk_text
from app.embeddings import embed
from app.extraction import extract_text_from_pdf

router = APIRouter(dependencies=[Depends(verify_api_key)])


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

    # embed() is CPU-bound, run in a thread so we don't block the async event loop
    vectors = await asyncio.to_thread(embed, raw_chunks)

    for idx, (chunk_text_, vector) in enumerate(zip(raw_chunks, vectors)):
        db.add(Chunk(
            document_id=doc.id,
            chunk_index=idx,
            text=chunk_text_,
            embedding=vector,
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

    if "pdf" in content_type or filename.endswith(".pdf"):
        try:
            text = await asyncio.to_thread(extract_text_from_pdf, raw_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"PDF extraction failed: {exc}",
            )
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
