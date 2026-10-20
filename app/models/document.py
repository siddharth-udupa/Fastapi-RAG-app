import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.config import settings
from app.core.db import Base


class Document(Base):
    """One uploaded file (the parent record)."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # One document -> many chunks.
    # `cascade="all, delete-orphan"` means deleting a Document also deletes
    # its chunks, so embeddings never outlive the file they came from.
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """
    One chunk of a document plus its embedding vector.

    This is the table pgvector searches. Each row is a "unit of meaning"
    that can be retrieved independently and pasted into the LLM prompt.

    Relationship: Document (1) -> DocumentChunk (many)
    """

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # position inside the document

    # The vector must have the same size as the embedding model's output.
    # For gemini-embedding-2 that is 3072 floats.
    embedding = Column(Vector(settings.EMBEDDING_DIMENSIONS), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")
