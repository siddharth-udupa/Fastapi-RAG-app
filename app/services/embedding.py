from google.genai import types

from app.core.config import settings
from app.core.llm import get_llm_client


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    """
    Small private helper: send texts to the embedding model, get vectors back.

    `task_type` tells Gemini what the vector will be used for:
    - RETRIEVAL_DOCUMENT -> text that will be stored and searched later
    - RETRIEVAL_QUERY    -> a question that searches the stored documents

    Using the matching task type for each side gives noticeably better
    search results than embedding both sides the same way.
    """
    if not texts:
        return []

    response = get_llm_client().models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )

    # response.embeddings is a list of objects; .values is the float vector.
    # The vectors come back in the same order as the input texts.
    return [embedding.values for embedding in response.embeddings]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed document chunks during ingestion.

    All chunks are sent in ONE API call. Batching is cheaper and faster
    than calling the API once per chunk.
    """
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(question: str) -> list[float]:
    """
    Embed a user's question at query time.

    Must use the same model as embed_documents(), otherwise the vectors
    live in different spaces and similarity search becomes meaningless.
    """
    vectors = _embed([question], task_type="RETRIEVAL_QUERY")
    return vectors[0]
