import logging
from collections.abc import Iterator

from google.genai import types

from app.core.config import settings
from app.core.llm import get_llm_client
from app.services.prompts import build_user_prompt, system_prompt
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def _generation_config(has_context: bool) -> types.GenerateContentConfig:
    """Shared settings for streaming and non-streaming calls."""
    return types.GenerateContentConfig(
        system_instruction=system_prompt(has_context),
        temperature=settings.LLM_TEMPERATURE,
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Non-streaming generation: wait for the full answer, then return it.

    Use this when you need the complete text (e.g. to save it, evaluate it,
    or return a JSON response with sources).
    """
    response = get_llm_client().models.generate_content(
        model=settings.LLM_MODEL,
        contents=build_user_prompt(question, chunks),
        config=_generation_config(has_context=bool(chunks)),
    )

    logger.info("generated answer with %d source(s)", len(chunks))
    return response.text or ""


def stream_answer(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    """
    Streaming generation: yield small text pieces as the model produces them.

    This function is a Python generator. It does not build a string and
    return it; it `yield`s tokens one at a time. FastAPI's StreamingResponse
    can send each token to the browser immediately.

    The LLM is called even when `chunks` is empty: the no-context system
    prompt tells it to answer from general knowledge and say so.
    """
    stream = get_llm_client().models.generate_content_stream(
        model=settings.LLM_MODEL,
        contents=build_user_prompt(question, chunks),
        config=_generation_config(has_context=bool(chunks)),
    )

    for event in stream:
        # Some events carry no text (e.g. metadata). Skip those.
        if event.text:
            yield event.text
