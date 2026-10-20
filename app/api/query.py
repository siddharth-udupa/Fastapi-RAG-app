import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_db
from app.schemas.query import QueryRequest, QueryResponse, SourceChunk
from app.services.rag import RagContext, RagPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# One pipeline for the whole app. The retriever is stateless, so sharing
# this object across requests is safe.
pipeline = RagPipeline()


def get_pipeline() -> RagPipeline:
    """
    FastAPI dependency that returns the app's RAG pipeline.

    Why a dependency instead of using the module-level object directly?
    - Tests can override it with `app.dependency_overrides[get_pipeline]`.
    - To use a hybrid retriever in production, change ONE line here:
          return RagPipeline(retriever=HybridRetriever())
      No endpoint code changes.
    """
    return pipeline


def _sse(event: str, data: dict) -> str:
    """
    Format one Server-Sent Event (SSE).

    SSE is a tiny text protocol:
        event: token
        data: {"text": "Hello"}

        <blank line ends the event>

    The blank line is required. Browsers and libraries use it as a delimiter.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_events(rag: RagPipeline, context: RagContext) -> Iterator[str]:
    """
    Stream the answer as SSE events:

      sources -> sent first, so the UI can render citations immediately
      token   -> one small piece of the answer, repeated many times
      done    -> marks the end of the stream
      error   -> sent if the LLM call fails mid-stream
    """
    yield _sse(
        "sources",
        {
            "sources": [
                SourceChunk.from_retrieved(chunk).model_dump(mode="json")
                for chunk in context.chunks
            ],
            "used_context": context.has_context,
        },
    )

    try:
        for token in rag.stream(context):
            yield _sse("token", {"text": token})
    except Exception:
        # Headers are already sent, so we cannot change the HTTP status code.
        # An `error` event lets the client show a useful message instead.
        logger.exception("LLM streaming failed")
        yield _sse("error", {"message": "The language model request failed."})
        return

    yield _sse("done", {"model": settings.LLM_MODEL})


def _prepare_context(
    rag: RagPipeline,
    request: QueryRequest,
    db: Session,
) -> RagContext:
    """
    Run retrieval for one request and convert infrastructure failures into
    a clear HTTP 502. This runs BEFORE streaming starts, so a normal error
    response is still possible here.
    """
    try:
        return rag.prepare(
            question=request.question,
            db=db,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=502,
            detail="Failed to retrieve relevant documents.",
        ) from exc


@router.post("", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
    rag: RagPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """
    Ask a question. Returns the full answer as JSON, plus sources.

    This endpoint is deliberately synchronous (`def`, not `async def`).
    The Gemini SDK and SQLAlchemy both make blocking calls. FastAPI runs
    sync endpoints in a worker thread, which keeps the event loop free.
    Marking it `async def` without `await`-friendly clients would block
    every other request while the LLM is thinking.
    """
    context = _prepare_context(rag, request, db)

    try:
        result = rag.answer(context)
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(
            status_code=502,
            detail="The language model request failed.",
        ) from exc

    return QueryResponse(
        answer=result.answer,
        sources=[SourceChunk.from_retrieved(c) for c in result.context.chunks],
        used_context=result.context.has_context,
        model=settings.LLM_MODEL,
    )


@router.post("/stream")
def stream_query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
    rag: RagPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """
    Ask a question and stream the answer token by token (SSE).

    Retrieval happens before the response starts, so the client receives
    the sources before the first token. If retrieval finds nothing, the
    LLM is still called with the no-context prompt.
    """
    context = _prepare_context(rag, request, db)

    return StreamingResponse(
        _stream_events(rag, context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx)
        },
    )
