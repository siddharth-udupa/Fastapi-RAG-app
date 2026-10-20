from app.services.retrieval import RetrievedChunk

# ---------------------------------------------------------------------------
# System prompts
#
# Keeping prompts in one module makes them easy to version, test and compare.
# This is the part of a RAG system you will change most often.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_WITH_CONTEXT = """You are a helpful assistant that answers questions
using the user's uploaded documents.

Rules:
1. Use ONLY the numbered sources in CONTEXT to answer.
2. Cite the sources you use like this: [Source 1].
3. If the context does not contain the answer, say:
   "I don't have enough information in the uploaded documents to answer that."
4. Never invent facts that are not supported by the context.
"""

SYSTEM_PROMPT_WITHOUT_CONTEXT = """You are a helpful assistant.

No relevant documents were found in the user's knowledge base for this question.
Answer from your general knowledge, but begin your answer with this exact sentence:

"No matching documents were found in your uploaded documents."

After that sentence, give a clear, concise answer. Do not cite sources, because
there are none. Make it obvious that this answer did not come from the user's files.
"""


def system_prompt(has_context: bool) -> str:
    """Pick the system prompt based on whether retrieval found anything."""
    return SYSTEM_PROMPT_WITH_CONTEXT if has_context else SYSTEM_PROMPT_WITHOUT_CONTEXT


def build_context_text(chunks: list[RetrievedChunk]) -> str:
    """
    Turn retrieved chunks into the CONTEXT block of the prompt.

    Each source is numbered so the LLM can cite it as [Source 1], [Source 2],
    and so you can map those citations back to real files later.
    """
    if not chunks:
        return ""

    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {number}] "
            f"(file: {chunk.filename}, chunk {chunk.chunk_index}, "
            f"similarity: {chunk.similarity:.3f})\n"
            f"{chunk.content}"
        )

    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Build the user message sent to the LLM.

    With context:    CONTEXT: ...  QUESTION: ...
    Without context: QUESTION: ...

    The "no context" branch is intentional. The LLM is still called so the
    user always gets an answer (and a clear note that no documents matched).
    """
    context = build_context_text(chunks)

    if not context:
        return f"QUESTION: {question}"

    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"
