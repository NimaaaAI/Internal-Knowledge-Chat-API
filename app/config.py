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

    class Config:
        env_file = ".env"


settings = Settings()
