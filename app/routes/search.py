import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.auth import verify_api_key
from app.database import get_db
from app.embeddings import embed_one
from app.schemas import SearchResponse, ChunkResult
from app.reranker import rerank_chunks
from app.config import settings

router = APIRouter(dependencies=[Depends(verify_api_key)])


def _build_where(
    doc_type: str | None,
    author: str | None,
    source: str | None,
) -> tuple[str, dict]:
    """Build a SQL WHERE clause for metadata filters."""
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if doc_type:
        clauses.append("d.doc_type = :doc_type")
        params["doc_type"] = doc_type
    if author:
        clauses.append("d.author = :author")
        params["author"] = author
    if source:
        clauses.append("d.source = :source")
        params["source"] = source

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


async def hybrid_search(
    db: AsyncSession,
    query: str,
    top_k: int,
    doc_type: str | None = None,
    author: str | None = None,
    source: str | None = None,
) -> list[ChunkResult]:
    """
    Hybrid search: vector similarity + full-text search, merged via RRF.

    RRF score = 1/(60 + rank_vector) + 1/(60 + rank_fts)
    The constant 60 dampens the outsized influence of rank-1 hits.
    A chunk that ranks well in BOTH searches scores higher than one
    that only dominates a single search method.
    """
    query_vec = await asyncio.to_thread(embed_one, query)
    where, params = _build_where(doc_type, author, source)

    params.update({
        "query_vec": str(query_vec),
        "ts_query": query,
        "top_k": top_k,
        "inner_limit": top_k * 4,  # oversample before merging
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
                 COALESCE(1.0 / (60 + f.rank), 0)) AS score
            FROM vector_ranked v
            FULL OUTER JOIN fts_ranked f ON v.id = f.id
        )
        SELECT
            c.id            AS chunk_id,
            c.document_id,
            c.chunk_index,
            c.text,
            rrf.score,
            rrf.vector_rank,
            rrf.fts_rank,
            d.title         AS document_title,
            d.source        AS document_source,
            d.author        AS document_author,
            d.doc_type,
            d.extra_metadata
        FROM rrf
        JOIN chunks c    ON c.id = rrf.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY rrf.score DESC
        LIMIT :top_k
    """)

    rows = (await db.execute(sql, params)).mappings().all()

    return [
        ChunkResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            score=float(row["score"]),
            document_title=row["document_title"],
            document_source=row["document_source"],
            document_author=row["document_author"],
            doc_type=row["doc_type"],
            extra_metadata=row["extra_metadata"] or {},
            vector_rank=row["vector_rank"],
            fts_rank=row["fts_rank"],
        )
        for row in rows
    ]


async def graph_expand(
    db: AsyncSession,
    base_chunks: list[ChunkResult],
    extra_k: int = 3,
) -> tuple[list[ChunkResult], list[dict]]:
    """
    Graph-based retrieval expansion.

    Finds the most-mentioned entities across the already-retrieved chunks,
    then fetches extra chunks from the DB that share those entities but
    weren't in the original results. The re-ranker downstream sorts the
    expanded pool by relevance — so graph expansion increases recall without
    sacrificing precision.
    """
    if not base_chunks:
        return base_chunks, []

    # UUIDs contain only hex + dashes — safe to inline
    chunk_id_list = "', '".join(str(c.chunk_id) for c in base_chunks)

    entity_rows = (await db.execute(text(f"""
        SELECT name, type, COUNT(*) AS freq
        FROM entities
        WHERE chunk_id IN ('{chunk_id_list}')
        GROUP BY name, type
        ORDER BY freq DESC
        LIMIT 5
    """))).mappings().all()

    if not entity_rows:
        return base_chunks, []

    top_entities = [{"name": r["name"], "type": r["type"]} for r in entity_rows]

    # Named params for entity names (user-controlled via Claude, must be parameterised)
    name_params = {f"en{i}": r["name"].lower() for i, r in enumerate(entity_rows)}
    name_in = ", ".join(f":en{i}" for i in range(len(entity_rows)))

    extra_rows = (await db.execute(text(f"""
        SELECT DISTINCT
            c.id            AS chunk_id,
            c.document_id,
            c.chunk_index,
            c.text,
            d.title         AS document_title,
            d.source        AS document_source,
            d.author        AS document_author,
            d.doc_type,
            d.extra_metadata
        FROM entities e
        JOIN chunks c    ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE lower(e.name) IN ({name_in})
          AND c.id NOT IN ('{chunk_id_list}')
        LIMIT :limit
    """), {**name_params, "limit": extra_k})).mappings().all()

    extra_chunks = [
        ChunkResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            score=0.0,          # no RRF score — sourced from entity graph
            document_title=row["document_title"],
            document_source=row["document_source"],
            document_author=row["document_author"],
            doc_type=row["doc_type"],
            extra_metadata=row["extra_metadata"] or {},
        )
        for row in extra_rows
    ]

    return base_chunks + extra_chunks, top_entities


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Natural-language search query"),
    top_k: int = Query(default=None, ge=1, le=50),
    doc_type: str | None = Query(default=None),
    author: str | None = Query(default=None),
    source: str | None = Query(default=None),
    rerank: bool = Query(default=False, description="Apply cross-encoder re-ranking after hybrid search"),
    db: AsyncSession = Depends(get_db),
):
    """Hybrid vector + full-text search with optional metadata filters and re-ranking."""
    if rerank:
        candidates = top_k or settings.rerank_candidates
        results = await hybrid_search(db, q, candidates, doc_type=doc_type, author=author, source=source)
        final_k = top_k or settings.rerank_top_k
        if len(results) > 1:
            results = await asyncio.to_thread(rerank_chunks, q, results, final_k)
    else:
        k = top_k or settings.search_top_k
        results = await hybrid_search(db, q, k, doc_type=doc_type, author=author, source=source)
    return SearchResponse(query=q, results=results)
