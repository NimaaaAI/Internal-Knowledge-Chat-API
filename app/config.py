from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/knowledge"

    # LLM — default is Claude, but any Anthropic model string works
    anthropic_api_key: str
    llm_model: str = "claude-sonnet-4-6"

    # Auth — leave empty to disable
    api_key: str = ""

    # Embedding model (sentence-transformers)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking
    chunk_words: int = 150       # words per chunk
    chunk_overlap_words: int = 30  # overlap between consecutive chunks

    # Retrieval
    search_top_k: int = 6

    # Re-ranking (cross-encoder applied after hybrid search)
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20   # fetch this many from hybrid search
    rerank_top_k: int = 6         # keep this many after re-ranking

    class Config:
        env_file = ".env"


settings = Settings()
