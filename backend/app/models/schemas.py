from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class WorkspaceSession:
    def __init__(self):
        self.document_ids: List[str] = []       # Active resource names
        self.active_docs: List[Document] = []   # List of loaded LangChain documents
        self.active_image: Optional[bytes] = None
        self.active_image_name: Optional[str] = None
        self.store = {}                         # Session chat history storage (InMemoryChatMessageHistory)
        self.active_quiz_data = None
        self.active_summary_data = None
        self.active_comparison_data = None
        self.messages = []                      # User/Assistant visual message logs
        self.is_processing = False              # Ingestion busy lock
        self.processing_message = ""

# Global session dictionary cache
sessions = {"default": WorkspaceSession()}

def get_session(session_id: str = "default") -> WorkspaceSession:
    if session_id not in sessions:
        sessions[session_id] = WorkspaceSession()
    return sessions[session_id]

# API Request/Response payload schemas
class ChatQuery(BaseModel):
    query: str
    session_id: str = "default"

class ActionRequest(BaseModel):
    session_id: str = "default"

class YouTubePayload(BaseModel):
    urls: str
    session_id: str = "default"

class RemoveSourcePayload(BaseModel):
    document_id: str
    session_id: str = "default"
