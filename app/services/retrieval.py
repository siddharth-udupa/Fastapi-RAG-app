import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.embedding import embed_query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """
    A chunk returned by the retriever, plus the metadata the API needs.

    A dataclass is just a small typed container. Using it instead of a plain
    dict means editors/type-checkers know exactly which fields exist.
    """

    chunk_id: str
    document_id: str
    filename: str
    content: str
    chunk_index: int
    similarity: float


def retrieve_chunks(
    question: str,
    db: Session,
    top_k: int = settings.TOP_K,
    min_similarity: float = settings.MIN_SIMILARITY,
) -> list[RetrievedChunk]:
    """
    Find the chunks that are most semantically similar to the question.

    The three steps of vector search:
    1. Embed the question with the SAME model used for the documents.
    2. Ask pgvector for the nearest vectors, using cosine distance (`<=>`).
    3. Keep the top_k results, optionally dropping weak matches.

    Cosine distance vs similarity:
    - `<=>` returns a distance: 0 = identical, larger = less similar.
    - `1 - distance` converts it to similarity: 1 = identical.
    We sort by distance ascending (equivalently similarity descending).

    If the table is empty, this simply returns an empty list. That is not an
    error: the pipeline will still call the LLM with no context.
    """
    query_vector = embed_query(question)

    # The CTE computes similarity once per row, so we can filter on it and
    # still ORDER BY it. `embedding IS NOT NULL` skips chunks whose embedding
    # job failed or is still pending.
    sql = text(
        """
        WITH ranked AS (
            SELECT
                c.id            AS chunk_id,
                c.document_id   AS document_id,
                d.filename      AS filename,
                c.content       AS content,
                c.chunk_index   AS chunk_index,
                1 - (c.embedding <=> CAST(:query_vector AS vector)) AS similarity
            FROM document_chunks AS c
            JOIN documents AS d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
        )
        SELECT *
        FROM ranked
        WHERE similarity >= :min_similarity
        ORDER BY similarity DESC
        LIMIT :top_k
        """
    )

    rows = db.execute(
        sql,
        {
            # pgvector accepts the text format: '[0.1, 0.2, ...]'
            "query_vector": str(query_vector),
            "min_similarity": min_similarity,
            "top_k": top_k,
        },
    ).fetchall()

    chunks = [
        RetrievedChunk(
            chunk_id=str(row.chunk_id),
            document_id=str(row.document_id),
            filename=row.filename,
            content=row.content,
            chunk_index=row.chunk_index,
            similarity=float(row.similarity),
        )
        for row in rows
    ]

    logger.info(
        "retrieved %d chunk(s) for question=%r (top_k=%d)",
        len(chunks),
        question,
        top_k,
    )
    return chunks
