from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration lives here and is loaded from the `.env` file.

    Pydantic validates these values once, at startup. If a required value
    (like DATABASE_URL) is missing, the app fails immediately instead of
    crashing later in the middle of a request.
    """

    # --- Required secrets / connections ---------------------------------
    DATABASE_URL: str
    GEMINI_API_KEY: str

    # --- Ingestion (how we split documents) -----------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- Retrieval (how many chunks we send to the LLM) -----------------
    TOP_K: int = 5

    # Chunks with a cosine similarity below this are thrown away.
    # 0.0 keeps everything. Raise it later to improve precision.
    MIN_SIMILARITY: float = 0.0

    # --- Models ----------------------------------------------------------
    # gemini-embedding-2 returns 3072-dimensional vectors.
    EMBEDDING_MODEL: str = "gemini-embedding-2"
    EMBEDDING_DIMENSIONS: int = 3072
    LLM_MODEL: str = "gemini-3.8-flash"

    # Low temperature = more factual / less creative. Good default for RAG.
    LLM_TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


# One shared Settings instance for the whole app.
# Import this everywhere instead of reading os.environ manually.
settings = Settings()
