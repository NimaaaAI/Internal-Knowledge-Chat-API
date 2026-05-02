import asyncio
import json

from app.llm import client
from app.config import settings

_PROMPT = """\
Extract named entities from the text below.
Return ONLY a JSON array — no explanation, no markdown code fence.
Each item must have exactly two keys: "name" (the entity as written) and "type" (one of: PERSON, ORGANIZATION, PLACE).
If no entities are found, return [].

Text:
{text}"""


async def extract_entities(text: str) -> list[dict]:
    """Call Claude to extract entities from one chunk. Returns [{name, type}, ...]."""
    try:
        msg = await client.messages.create(
            model=settings.llm_model,
            max_tokens=400,
            messages=[{"role": "user", "content": _PROMPT.format(text=text[:2000])}],
        )
        raw = msg.content[0].text.strip()
        # Strip markdown code fences if Claude adds them
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        entities = json.loads(raw)
        if not isinstance(entities, list):
            return []
        return [
            {"name": str(e["name"]), "type": str(e["type"])}
            for e in entities
            if isinstance(e, dict)
            and "name" in e
            and e.get("type") in ("PERSON", "ORGANIZATION", "PLACE")
        ]
    except Exception:
        return []


async def extract_all_entities(texts: list[str], concurrency: int = 5) -> list[list[dict]]:
    """Extract entities from multiple chunks concurrently (rate-limited by semaphore)."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(text: str) -> list[dict]:
        async with sem:
            return await extract_entities(text)

    return list(await asyncio.gather(*[_one(t) for t in texts]))
