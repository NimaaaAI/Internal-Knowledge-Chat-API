import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.auth import verify_api_key
from app.database import get_db
from app.models import Document, Chunk
from app.schemas import DocumentOut

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db)):
    """Return all uploaded documents with their chunk count."""
    rows = await db.execute(
        select(
            Document,
            func.count(Chunk.id).label("chunk_count"),
        )
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    )

    results = []
    for doc, chunk_count in rows.all():
        results.append(DocumentOut(
            id=doc.id,
            title=doc.title,
            source=doc.source,
            author=doc.author,
            doc_type=doc.doc_type,
            content_type=doc.content_type,
            extra_metadata=doc.extra_metadata,
            created_at=doc.created_at,
            chunk_count=chunk_count,
        ))
    return results


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Delete a document and all its chunks (cascade handled by DB foreign key)."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    await db.delete(doc)
    await db.commit()
