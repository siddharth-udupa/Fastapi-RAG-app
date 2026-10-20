from functools import lru_cache

from google import genai

from app.core.config import settings


@lru_cache(maxsize=1)
def get_llm_client() -> genai.Client:
    """
    Return the one shared Gemini client for the whole application.

    Why cache it?
    - Creating a client opens HTTP connection pools and re-reads credentials.
      Doing that on every request wastes time.
    - `lru_cache(maxsize=1)` means: call this function as often as you like,
      but the body only runs once. Every later call returns the same object.

    Both embeddings (`services/embedding.py`) and chat completions
    (`services/generation.py`) use this same client.
    """
    return genai.Client(api_key=settings.GEMINI_API_KEY)
