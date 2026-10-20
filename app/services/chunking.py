from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

def chunk_text(text: str) -> list[str]:
    """
    Split a long document text into smaller overlapping chunks.
    
    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraph breaks first, then sentences, then words,
    then characters — so chunks are semantically cleaner than naive fixed-size splits.
    
    Why overlap?
    If a key sentence falls at the boundary of two chunks, overlap ensures
    it appears in at least one chunk fully, preventing missed context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)