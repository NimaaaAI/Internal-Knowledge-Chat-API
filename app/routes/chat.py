import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import anthropic

from app.auth import verify_api_key
from app.database import get_db
from app.schemas import ChatRequest, ChatResponse, SourceReference
from app.routes.search import hybrid_search
from app.config import settings

router = APIRouter(dependencies=[Depends(verify_api_key)])

_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """\
You are a knowledgeable assistant with access to an internal document library.
Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information, say so clearly — do not invent facts.
When you use information from a passage, note it inline as [Doc: <title>, chunk <index>].
Be concise and precise."""


def _build_context(chunks) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Doc: {c.document_title}, chunk {c.chunk_index}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: AsyncSession = Depends(get_db)):
    chunks = await hybrid_search(
        db,
        payload.question,
        top_k=settings.search_top_k,
        doc_type=payload.doc_type,
        author=payload.author,
        source=payload.source,
    )

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
