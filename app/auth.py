from fastapi import Header, HTTPException, status
from app.config import settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If API_KEY is set in .env, every request must include X-Api-Key: <key>."""
    if not settings.api_key:
        return  # auth disabled — fine for local dev
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
