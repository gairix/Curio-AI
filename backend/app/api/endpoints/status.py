from fastapi import APIRouter
from backend.app.models.schemas import get_session

router = APIRouter()

@router.get("/status")
async def get_status(session_id: str = "default"):
    ws = get_session(session_id)
    return {
        "document_ids": ws.document_ids,
        "has_active_image": ws.active_image is not None,
        "active_image_name": ws.active_image_name,
        "message_count": len(ws.messages),
        "is_processing": ws.is_processing,
        "processing_message": ws.processing_message,
        "has_summary": ws.active_summary_data is not None,
        "has_quiz": ws.active_quiz_data is not None,
        "has_comparison": ws.active_comparison_data is not None,
    }
