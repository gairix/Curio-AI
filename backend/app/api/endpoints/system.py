from fastapi import APIRouter

from backend.app.models.schemas import get_session, ActionRequest, RemoveSourcePayload

router = APIRouter()

@router.post("/reset")
async def reset_workspace(payload: ActionRequest):
    """Wipe all active PDFs, transcriptions, images, chat history, and evaluation state."""
    ws = get_session(payload.session_id)
    ws.messages = []
    ws.document_ids = []
    ws.active_docs = []
    ws.active_image = None
    ws.active_image_name = None
    ws.active_quiz_data = None
    ws.active_summary_data = None
    ws.active_comparison_data = None
    if "user_1" in ws.store:
        del ws.store["user_1"]
    return {"status": "success", "message": "Workspace reset complete!"}

@router.post("/remove-source")
async def remove_source(payload: RemoveSourcePayload):
    """Remove a specific document from active documents and session tracking."""
    ws = get_session(payload.session_id)
    doc_id = payload.document_id
    
    # Remove from active resource name tracking
    ws.document_ids = [d for d in ws.document_ids if d != doc_id]
    
    # Filter active_docs to remove chunks belonging to this document
    ws.active_docs = [doc for doc in ws.active_docs if doc.metadata.get("document_id") != doc_id]
    
    # If the active image matches the removed resource name, clear the image
    if ws.active_image_name == doc_id:
        ws.active_image = None
        ws.active_image_name = None
        
    # Also reset quiz/summary/comparison if no documents are left
    if not ws.document_ids:
        ws.active_quiz_data = None
        ws.active_summary_data = None
        ws.active_comparison_data = None
        
    return {
        "status": "success",
        "message": f"Successfully removed '{doc_id}' from the workspace!"
    }
