import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.schemas.document import DocumentUploadResponse
from app.services.ingestion import ingest_document

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Upload a PDF. The file is parsed, chunked, embedded and stored.

    `UploadFile = File(...)` is FastAPI's file-upload support, roughly the
    equivalent of multer in Express.
    `Depends(get_db)` is dependency injection: FastAPI creates a DB session,
    passes it here, and closes it when the request finishes.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # This endpoint is sync (`def`), so we cannot `await file.read()`.
    # `file.file` is the underlying SpooledTemporaryFile and supports a
    # normal blocking .read(). FastAPI runs this in a worker thread.
    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        result = ingest_document(
            filename=file.filename,
            file_bytes=file_bytes,
            db=db,
        )
    except ValueError as exc:
        # Expected, user-fixable problems: scanned PDF, no text, etc.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the document.",
        ) from exc

    return DocumentUploadResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        chunks_created=result["chunks_created"],
        message=f"Successfully ingested {result['chunks_created']} chunks.",
    )
