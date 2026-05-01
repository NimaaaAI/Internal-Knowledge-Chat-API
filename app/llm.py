import anthropic
from app.config import settings

client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """\
You are a knowledgeable assistant with access to an internal document library.
Answer the user's question using ONLY the provided context passages.
If the context does not contain enough information, say so clearly — do not invent facts.
When you use information from a passage, note it inline as [Doc: <title>, chunk <index>].
Be concise and precise."""


def build_context(chunks) -> str:
    parts = [f"[Doc: {c.document_title}, chunk {c.chunk_index}]\n{c.text}" for c in chunks]
    return "\n\n---\n\n".join(parts)
