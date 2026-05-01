from sentence_transformers import CrossEncoder

from app.config import settings

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.rerank_model)
    return _reranker


def rerank_chunks(query: str, chunks: list, top_k: int) -> list:
    """
    Score each (query, chunk) pair with the cross-encoder and return
    the top_k chunks sorted by score descending.

    The cross-encoder reads query and passage *together* (unlike the
    bi-encoder used for embeddings), so it understands their interaction
    directly — this is why it's more precise but slower than vector search.
    """
    if not chunks:
        return chunks
    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs).tolist()
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:top_k]]


def rerank_with_scores(query: str, chunks: list) -> list[tuple[float, object]]:
    """Return (score, chunk) pairs sorted by score descending (used by the trace endpoint)."""
    if not chunks:
        return []
    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs).tolist()
    return sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
