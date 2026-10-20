import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="RAG App",
    description="A small, readable RAG system built with FastAPI + pgvector + Gemini.",
    version="0.2.0",
)

# API routes first...
app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# ...then the static frontend.
# Order matters: app.mount("/", ...) catches EVERY path, so mounting it
# before the API routes would hide them. Keep the catch-all mount last.
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
