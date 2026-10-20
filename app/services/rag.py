from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.generation import generate_answer, stream_answer
from app.services.retrieval import RetrievedChunk, retrieve_chunks

# ---------------------------------------------------------------------------
# The retriever contract
#
# A "retriever" is any function that can turn a question into ranked chunks.
# Today there is exactly one: `retrieve_chunks` (pure vector search).
#
# Later you can add hybrid search, MMR, a parent-document retriever, etc.
# As long as the new function accepts the same arguments and returns
# list[RetrievedChunk], you can plug it into RagPipeline without touching
# the API layer or the generation code.
# ---------------------------------------------------------------------------
class Retriever(Protocol):
    def __call__(
        self,
        *,
        question: str,
        db: Session,
        top_k: int,
        min_similarity: float,
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class RagContext:
    """Everything retrieval produced for one question."""

    question: str
    chunks: list[RetrievedChunk]

    @property
    def has_context(self) -> bool:
        return len(self.chunks) > 0


@dataclass(frozen=True)
class RagAnswer:
    """A finished (non-streaming) answer and the context that produced it."""

    answer: str
    context: RagContext


class RagPipeline:
    """
    The complete RAG workflow, split into two readable stages:

        prepare()  -> embed the question + retrieve chunks
        answer()   -> build the prompt + call the LLM (non-streaming)
        stream()   -> build the prompt + call the LLM (streaming)

    Why a class instead of one big function?
    - The retriever is injected in the constructor, so it can be swapped
      for a better one in tests or in a future hybrid-search version.
    - Advanced stages (query rewriting, reranking, filtering) belong inside
      `prepare()`. Nothing else in the app needs to change when you add them.

    Why two dataclasses?
    - `RagContext` is retrieval output. It can be inspected, logged, cached,
      or fed to an evaluator without calling the LLM.
    - `RagAnswer` keeps the answer and its sources together, which is what
      the API needs to build a response with citations.
    """

    def __init__(
        self,
        retriever: Retriever = retrieve_chunks,
        min_similarity: float = settings.MIN_SIMILARITY,
    ) -> None:
        self._retriever = retriever
        self._min_similarity = min_similarity

    def prepare(
        self,
        question: str,
        db: Session,
        top_k: int = settings.TOP_K,
    ) -> RagContext:
        """
        Stage 1: question -> embedding -> vector search -> chunks.

        Add advanced retrieval here later:
          1. rewrite the question into better search queries
          2. retrieve a wider candidate set
          3. rerank with a cross-encoder
          4. compress / deduplicate chunks
          5. build the final RagContext
        """
        chunks = self._retriever(
            question=question,
            db=db,
            top_k=top_k,
            min_similarity=self._min_similarity,
        )
        return RagContext(question=question, chunks=chunks)

    def answer(self, context: RagContext) -> RagAnswer:
        """Stage 2: context -> prompt -> full LLM answer."""
        text = generate_answer(context.question, context.chunks)
        return RagAnswer(answer=text, context=context)

    def stream(self, context: RagContext) -> Iterator[str]:
        """Stage 2 (streaming): context -> prompt -> tokens."""
        return stream_answer(context.question, context.chunks)
