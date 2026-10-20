from fastapi import APIRouter

from app.api import documents, query

# This is the API v1 router. main.py mounts it at /api/v1.
router = APIRouter()

# Final paths:
#   POST /api/v1/documents/upload
#   POST /api/v1/query
#   POST /api/v1/query/stream
router.include_router(documents.router, prefix="/documents", tags=["Documents"])
router.include_router(query.router, prefix="/query", tags=["Query"])
