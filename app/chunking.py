import re
from app.config import settings


def chunk_text(text: str) -> list[str]:
    """
    Word-based sliding-window chunker.

    Why word-based: all-MiniLM-L6-v2 has a 256 wordpiece token limit.
    150 English words ≈ 210 wordpieces — safe headroom.
    30-word overlap prevents losing meaning at chunk boundaries.
    """
    size = settings.chunk_words # 150 words per chunk
    overlap = settings.chunk_overlap_words # 30 words overlap between chunks

    # Split on whitespace, keeping the separators so we can reconstruct spacing
    tokens = re.split(r"(\s+)", text.strip())

    # words[i] = (original_token_index, word_string) — only non-whitespace tokens
    words = [(i, t) for i, t in enumerate(tokens) if t.strip()]

    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + size, len(words))

        first_token_idx = words[start][0]
        last_token_idx = words[end - 1][0]

        chunk = "".join(tokens[first_token_idx: last_token_idx + 1]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start += size - overlap

    return chunks
