from sentence_transformers import SentenceTransformer
from app.config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns L2-normalised vectors (required for cosine similarity)."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
