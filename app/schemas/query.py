from uuid import UUID

from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.retrieval import RetrievedChunk


class QueryRequest(BaseModel):
    """The JSON body the client sends to ask a question."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The question to answer from the uploaded documents.",
    )
    top_k: int = Field(
        default=settings.TOP_K,
        ge=1,
        le=20,
        description="How many chunks to retrieve before calling the LLM.",
    )


class SourceChunk(BaseModel):
    """One retrieved chunk, returned so the UI can show its citations."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    chunk_index: int
    similarity: float
    content: str

    @classmethod
    def from_retrieved(cls, chunk: RetrievedChunk) -> "SourceChunk":
        """Adapter: service dataclass -> API schema."""
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            similarity=chunk.similarity,
            content=chunk.content,
        )


class QueryResponse(BaseModel):
    """Non-streaming response: the answer plus the sources used to produce it."""

    answer: str
    sources: list[SourceChunk]
    used_context: bool
    model: str
