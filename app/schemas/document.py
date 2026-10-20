from pydantic import BaseModel
from uuid import UUID
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    """What the API returns after a successful upload."""
    document_id: UUID
    filename: str
    chunks_created: int
    message: str