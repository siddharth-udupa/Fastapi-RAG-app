import io
from typing import TypedDict

from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentChunk
from app.services.chunking import chunk_text
from app.services.embedding import embed_documents


class IngestionResult(TypedDict):
    """What `ingest_document` returns to the API layer."""

    document_id: str
    filename: str
    chunks_created: int


def parse_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF into one string, page by page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    # Double newline keeps page/paragraph boundaries intact for the splitter.
    return "\n\n".join(pages)


def ingest_document(filename: str, file_bytes: bytes, db: Session) -> IngestionResult:
    """
    Full ingestion pipeline:

    PDF bytes
      -> text          (parse_pdf)
      -> chunks        (chunk_text)
      -> vectors       (embed_documents, one batch call)
      -> database      (Document + DocumentChunk rows)

    The embedding call happens BEFORE any database write, so a failed API
    call leaves the database untouched. The commit is wrapped in a
    try/except so a failed insert rolls back cleanly.
    """
    try:
        raw_text = parse_pdf(file_bytes)
    except Exception as exc:
        raise ValueError("Could not read this PDF. It may be corrupted.") from exc

    if not raw_text.strip():
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned/image-only document."
        )

    chunks = chunk_text(raw_text)
    if not chunks:
        raise ValueError("The document produced no text chunks.")

    embeddings = embed_documents(chunks)

    try:
        document = Document(filename=filename)
        db.add(document)
        # flush() sends the INSERT and gives us document.id, without
        # committing yet. We need that id for the chunks' foreign key.
        db.flush()

        chunk_rows = [
            DocumentChunk(
                document_id=document.id,
                content=chunk,
                chunk_index=index,
                embedding=embedding,
            )
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        db.add_all(chunk_rows)
        db.commit()
    except Exception:
        db.rollback()
        raise

    return {
        "document_id": str(document.id),
        "filename": filename,
        "chunks_created": len(chunk_rows),
    }
