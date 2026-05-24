from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import get_session, ActionRequest
from backend.app.services.llm import (
    generate_summary_service,
    generate_quiz_service,
    generate_comparison_service
)

router = APIRouter()

@router.post("/action/summary")
async def generate_summary(payload: ActionRequest):
    """Generate high-density academic summary themes over the active resources."""
    ws = get_session(payload.session_id)
    if not ws.document_ids:
        raise HTTPException(status_code=400, detail="No active resources loaded.")
        
    try:
        summary_data = await generate_summary_service(ws)
        return {"summary": summary_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/action/quiz")
async def generate_quiz(payload: ActionRequest):
    """Generate interactive multiple choice questions based on the active resources."""
    ws = get_session(payload.session_id)
    if not ws.document_ids:
        raise HTTPException(status_code=400, detail="No active resources loaded.")
        
    try:
        quiz_data = await generate_quiz_service(ws)
        return {"quiz": quiz_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/action/compare")
async def generate_comparison(payload: ActionRequest):
    """Generate comparative matrix grids over active textual resources."""
    ws = get_session(payload.session_id)
    if not ws.document_ids:
        raise HTTPException(status_code=400, detail="No active resources loaded.")
        
    try:
        comparison_data = await generate_comparison_service(ws)
        return {"comparison": comparison_data}
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
