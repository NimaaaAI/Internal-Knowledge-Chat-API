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
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text.strip()) if p.strip()]

    if not paragraphs:
        return []

    def word_count(s: str) -> int:
        return len(s.split())

    def split_sentences(paragraph: str) -> list[str]:
        # Fallback: split a long paragraph into sentences
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]

    # Build a flat list of units — either whole paragraphs or sentences
    # (sentences only appear when a paragraph exceeds the word limit)
    units: list[str] = []
    for para in paragraphs:
        if word_count(para) <= size:
            units.append(para)
        else:
            units.extend(split_sentences(para))

    # Group units into chunks up to the word limit
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for unit in units:
        uw = word_count(unit)
        if current and current_words + uw > size:
            chunks.append("\n\n".join(current))
            # Overlap: carry last unit into the next chunk
            current = [current[-1]]
            current_words = word_count(current[0])
        current.append(unit)
        current_words += uw

    if current:
        chunks.append("\n\n".join(current))

    return chunks
    