from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.embeddings import get_model
from app.routes import upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model once at startup so the first request is not slow
    get_model()
    yield


app = FastAPI(
    title="Internal Knowledge Chat API",
    description="Upload documents, search by meaning, chat with your knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(upload.router, tags=["Upload"])

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
