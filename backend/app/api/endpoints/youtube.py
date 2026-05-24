import re
from datetime import datetime
from fastapi import APIRouter, HTTPException

from backend.app.models.schemas import get_session, YouTubePayload
from backend.app.services.ingestion import ingest_youtube
from backend.app.utils.helpers import extract_yt_identifier
from backend.app.services.retrieval import build_vectorstore

router = APIRouter()

@router.post("/process/youtube")
async def process_youtube(payload: YouTubePayload):
    """Ingest YouTube URL transcripts, chunk their narratives and write embeddings to Pinecone."""
    ws = get_session(payload.session_id)
    ws.is_processing = True
    ws.processing_message = "Ingesting YouTube video details..."
    
    try:
        # Split by comma, semicolon, newline, carriage return, or double-plus spaces
        urls = [u.strip() for u in re.split(r'[,\n\r;]|\s{2,}', payload.urls) if u.strip()]
        all_processed_docs = []
        source_ids_list = []
        
        for url in urls:
            if "youtube.com" in url or "youtu.be" in url:
                target_vid_id = extract_yt_identifier(url)
                if not target_vid_id:
                    continue
                    
                file_docs = ingest_youtube(url, target_vid_id)
                if file_docs:
                    source_ids_list.append(target_vid_id)
                    all_processed_docs.extend(file_docs)
            else:
                pass  # Skip non-YouTube urls
                
        if all_processed_docs:
            ws.active_docs.extend(all_processed_docs)
            ws.document_ids.extend(source_ids_list)
            ws.active_image = None
            ws.active_image_name = None
            
            # Sync to Pinecone index
            vstore = build_vectorstore()
            vstore.add_documents(
                all_processed_docs,
                ids=[f"{i}_{datetime.now().timestamp()}" for i in range(len(all_processed_docs))]
            )
            
        ws.is_processing = False
        return {
            "status": "success",
            "message": f"Indexed {len(all_processed_docs)} video transcript segments!",
            "document_ids": ws.document_ids
        }
    except Exception as e:
        ws.is_processing = False
        raise HTTPException(status_code=500, detail=str(e))
