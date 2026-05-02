import asyncio
import json
import time
import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, text

from app.auth import verify_api_key
from app.config import settings
from app.database import get_db
from app.embeddings import embed_one
from app.llm import client as _llm_client, SYSTEM_PROMPT, build_context
from app.models import Chunk, Document
from app.reranker import rerank_with_scores
from app.routes.search import hybrid_search, graph_expand, _build_where
from app.schemas import ChunkDetail, DebugStats, TraceResponse, TraceResult

router = APIRouter(prefix="/debug", dependencies=[Depends(verify_api_key)])


@router.get("/stats", response_model=DebugStats)
async def debug_stats(db: AsyncSession = Depends(get_db)):
    doc_count = await db.scalar(select(func.count(Document.id)))
    chunk_count = await db.scalar(select(func.count(Chunk.id)))
    return DebugStats(document_count=doc_count or 0, chunk_count=chunk_count or 0)


@router.get("/chunks", response_model=list[ChunkDetail])
async def debug_chunks(
    document_id: uuid.UUID = Query(...),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            text("""
                SELECT c.id, c.document_id, c.chunk_index, c.text,
                       vector_dims(c.embedding) AS embedding_dims,
                       d.title AS document_title
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id = :doc_id
                ORDER BY c.chunk_index
            """),
            {"doc_id": str(document_id)},
        )
    ).mappings().all()

    return [
        ChunkDetail(
            chunk_id=row["id"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            embedding_dims=row["embedding_dims"],
        )
        for row in rows
    ]


@router.get("/trace", response_model=TraceResponse)
async def debug_trace(
    q: str = Query(..., min_length=1),
    top_k: int = Query(default=10, ge=1, le=50),
    doc_type: str | None = Query(default=None),
    author: str | None = Query(default=None),
    source: str | None = Query(default=None),
    rerank: bool = Query(default=False, description="Apply cross-encoder re-ranking and show scores"),
    db: AsyncSession = Depends(get_db),
):
    query_vec = await asyncio.to_thread(embed_one, q)
    where, params = _build_where(doc_type, author, source)

    params.update({
        "query_vec": str(query_vec),
        "ts_query": q,
        "top_k": top_k,
        "inner_limit": top_k * 4,
    })

    sql = text(f"""
        WITH filtered AS (
            SELECT c.id, c.document_id, c.chunk_index, c.text,
                   c.embedding, c.ts_vector
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            {where}
        ),
        vector_ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:query_vec AS vector)) AS rank
            FROM filtered
            ORDER BY embedding <=> CAST(:query_vec AS vector)
            LIMIT :inner_limit
        ),
        fts_ranked AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank(ts_vector, plainto_tsquery('english', :ts_query)) DESC
                   ) AS rank
            FROM filtered
            WHERE ts_vector @@ plainto_tsquery('english', :ts_query)
            LIMIT :inner_limit
        ),
        rrf AS (
            SELECT
                COALESCE(v.id, f.id) AS id,
                v.rank               AS vector_rank,
                f.rank               AS fts_rank,
                (COALESCE(1.0 / (60 + v.rank), 0) +
                 COALESCE(1.0 / (60 + f.rank), 0)) AS rrf_score
            FROM vector_ranked v
            FULL OUTER JOIN fts_ranked f ON v.id = f.id
        )
        SELECT
            c.id            AS chunk_id,
            c.document_id,
            c.chunk_index,
            c.text,
            rrf.vector_rank,
            rrf.fts_rank,
            rrf.rrf_score,
            d.title         AS document_title
        FROM rrf
        JOIN chunks c    ON c.id = rrf.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY rrf.rrf_score DESC
        LIMIT :top_k
    """)

    rows = (await db.execute(sql, params)).mappings().all()

    results = [
        TraceResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            vector_rank=row["vector_rank"],
            fts_rank=row["fts_rank"],
            rrf_score=float(row["rrf_score"]),
        )
        for row in rows
    ]

    if rerank and len(results) > 1:
        scored_pairs = await asyncio.to_thread(rerank_with_scores, q, results)
        results = [
            r.model_copy(update={"rerank_score": float(score)})
            for score, r in scored_pairs[:top_k]
        ]

    return TraceResponse(query=q, results=results)


# ── Entity Graph ─────────────────────────────────────────────────────────────

@router.get("/entities")
async def debug_entities(
    name: str = Query(default="", description="Partial name to search"),
    type: str | None = Query(default=None, description="PERSON, ORGANIZATION, or PLACE"),
    limit: int = Query(default=30, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Search extracted entities across all chunks."""
    conditions, params = [], {"limit": limit}
    if name:
        conditions.append("lower(e.name) LIKE :name_pat")
        params["name_pat"] = f"%{name.lower()}%"
    if type:
        conditions.append("e.type = :type")
        params["type"] = type
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    rows = (await db.execute(text(f"""
        SELECT e.name, e.type, d.title AS document_title,
               d.id AS document_id, c.id AS chunk_id, c.chunk_index,
               left(c.text, 160) AS text_preview
        FROM entities e
        JOIN chunks c    ON c.id = e.chunk_id
        JOIN documents d ON d.id = e.document_id
        {where}
        ORDER BY lower(e.name), d.title
        LIMIT :limit
    """), params)).mappings().all()

    return [dict(r) for r in rows]


@router.get("/cooccurrence")
async def debug_cooccurrence(
    name: list[str] = Query(..., description="Two or more entity names to find together"),
    db: AsyncSession = Depends(get_db),
):
    """
    Find chunks that mention ALL given entities — the core graph query.
    Example: ?name=Jane+Smith&name=Nordic shows chunks where both appear.
    """
    if len(name) < 1:
        return []
    name_params = {f"n{i}": nm.lower() for i, nm in enumerate(name)}
    name_in = ", ".join(f":n{i}" for i in range(len(name)))

    rows = (await db.execute(text(f"""
        SELECT c.id AS chunk_id, c.document_id, c.chunk_index,
               left(c.text, 300) AS text_preview,
               d.title AS document_title, d.doc_type, d.author
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN (
            SELECT chunk_id FROM entities
            WHERE lower(name) IN ({name_in})
            GROUP BY chunk_id
            HAVING COUNT(DISTINCT lower(name)) = :name_count
        )
        LIMIT 20
    """), {**name_params, "name_count": len(name)})).mappings().all()

    return [dict(r) for r in rows]


# ── Pipeline Inspector ────────────────────────────────────────────────────────

def _evt(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _pipeline_stream(q: str, doc_type: str | None, author: str | None, db: AsyncSession):
    # Step 1 — query received
    yield _evt({"step": "query", "status": "done", "data": {
        "question": q,
        "filters": {"doc_type": doc_type, "author": author},
    }})

    # Step 2 — hybrid search
    yield _evt({"step": "hybrid_search", "status": "running", "data": {}})
    t0 = time.perf_counter()
    candidates = await hybrid_search(
        db, q,
        top_k=settings.rerank_candidates if settings.rerank_enabled else settings.search_top_k,
        doc_type=doc_type,
        author=author,
    )
    hs_ms = int((time.perf_counter() - t0) * 1000)

    yield _evt({"step": "hybrid_search", "status": "done", "data": {
        "total_candidates": len(candidates),
        "duration_ms": hs_ms,
        "top_chunks": [
            {
                "document_title": c.document_title,
                "chunk_index": c.chunk_index,
                "rrf_score": round(c.score, 5),
                "vector_rank": c.vector_rank,
                "fts_rank": c.fts_rank,
                "text_preview": c.text[:130] + ("…" if len(c.text) > 130 else ""),
            }
            for c in candidates[:8]
        ],
    }})

    if not candidates:
        yield _evt({"step": "error", "data": {"message": "No documents found for this question."}})
        yield "data: [DONE]\n\n"
        return

    # Step 3 — graph expansion (only if enabled)
    if settings.graph_enabled:
        yield _evt({"step": "graph", "status": "running", "data": {}})
        t0 = time.perf_counter()
        expanded, top_entities = await graph_expand(db, candidates, extra_k=settings.graph_extra_chunks)
        graph_ms = int((time.perf_counter() - t0) * 1000)
        extra_count = len(expanded) - len(candidates)

        yield _evt({"step": "graph", "status": "done", "data": {
            "duration_ms": graph_ms,
            "entities_used": top_entities,
            "extra_chunks_added": extra_count,
            "total_pool": len(expanded),
            "extra_chunks": [
                {
                    "document_title": c.document_title,
                    "chunk_index": c.chunk_index,
                    "text_preview": c.text[:130] + ("…" if len(c.text) > 130 else ""),
                }
                for c in expanded[len(candidates):]
            ],
        }})
        candidates = expanded

    # Step 4 — re-ranking (only if enabled)
    if settings.rerank_enabled and len(candidates) > 1:
        yield _evt({"step": "reranking", "status": "running", "data": {}})
        t0 = time.perf_counter()
        scored_pairs = await asyncio.to_thread(rerank_with_scores, q, candidates)
        rr_ms = int((time.perf_counter() - t0) * 1000)

        original_rank = {str(c.chunk_id): i + 1 for i, c in enumerate(candidates)}
        reranked_preview = []
        for new_rank, (score, chunk) in enumerate(scored_pairs[:settings.rerank_top_k], start=1):
            orig = original_rank.get(str(chunk.chunk_id), new_rank)
            reranked_preview.append({
                "document_title": chunk.document_title,
                "chunk_index": chunk.chunk_index,
                "rerank_score": round(float(score), 4),
                "original_rank": orig,
                "new_rank": new_rank,
                "moved": orig - new_rank,
                "text_preview": chunk.text[:130] + ("…" if len(chunk.text) > 130 else ""),
            })

        final_chunks = [chunk for _, chunk in scored_pairs[:settings.rerank_top_k]]

        yield _evt({"step": "reranking", "status": "done", "data": {
            "duration_ms": rr_ms,
            "top_chunks": reranked_preview,
        }})
    else:
        final_chunks = candidates[:settings.search_top_k]

    # Step 4 — context assembly
    context = build_context(final_chunks)
    word_count = len(context.split())
    yield _evt({"step": "context", "status": "done", "data": {
        "chunk_count": len(final_chunks),
        "word_count": word_count,
        "context_preview": context[:600] + ("…" if len(context) > 600 else ""),
    }})

    # Step 5 — LLM streaming
    yield _evt({"step": "llm", "status": "running", "data": {}})

    user_message = f"Context:\n\n{context}\n\nQuestion: {q}"
    full_answer = ""

    async with _llm_client.messages.stream(
        model=settings.llm_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for delta in stream.text_stream:
            full_answer += delta
            yield _evt({"step": "llm_token", "data": {"token": delta}})

    sources = [
        {"document_title": c.document_title, "chunk_index": c.chunk_index}
        for c in final_chunks
    ]
    yield _evt({"step": "done", "data": {"answer": full_answer, "sources": sources}})
    yield "data: [DONE]\n\n"


@router.get("/pipeline")
async def debug_pipeline(
    q: str = Query(..., min_length=1),
    doc_type: str | None = Query(default=None),
    author: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    return StreamingResponse(
        _pipeline_stream(q, doc_type, author, db),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
