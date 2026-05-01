import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import verify_api_key
from app.database import get_db
from app.llm import client as _client, SYSTEM_PROMPT, build_context as _build_context
from app.schemas import ChatRequest, ChatResponse, SourceReference
from app.routes.search import hybrid_search
from app.reranker import rerank_chunks
from app.config import settings

router = APIRouter(dependencies=[Depends(verify_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    candidate_k = settings.rerank_candidates if settings.rerank_enabled else settings.search_top_k
    chunks = await hybrid_search(
        db,
        payload.question,
        top_k=candidate_k,
        doc_type=payload.doc_type,
        author=payload.author,
        source=payload.source,
    )

    if settings.rerank_enabled and len(chunks) > 1:
        chunks = await asyncio.to_thread(rerank_chunks, payload.question, chunks, settings.rerank_top_k)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant documents found for this question.",
        )

    context = _build_context(chunks)
    user_message = f"Context:\n\n{context}\n\nQuestion: {payload.question}"

    if payload.stream:
        return StreamingResponse(
            _stream(user_message, chunks),
            media_type="text/event-stream",
        )

    message = await _client.messages.create(
        model=settings.llm_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    sources = [
        SourceReference(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_index=c.chunk_index,
            text=c.text,
        )
        for c in chunks
    ]

    return ChatResponse(answer=message.content[0].text, sources=sources)


async def _stream(user_message: str, chunks):
    """Server-Sent Events: stream answer tokens, then send sources as the last event."""
    sources = [
        SourceReference(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_index=c.chunk_index,
            text=c.text,
        ).model_dump(mode="json")
        for c in chunks
    ]

    async with _client.messages.stream(
        model=settings.llm_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for delta in stream.text_stream:
            yield f"data: {json.dumps({'delta': delta})}\n\n"

    yield f"data: {json.dumps({'sources': sources})}\n\n"
    yield "data: [DONE]\n\n"
