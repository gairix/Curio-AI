import os
import shutil
import asyncio
from datetime import datetime
from typing import List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from backend.app.config import DOWNLOADS_DIR
from backend.app.models.schemas import get_session
from backend.app.core.models import whisper_model
from backend.app.utils.helpers import create_timestamped_chunks
from backend.app.services.ingestion import ingest_pdf, ingest_local_media
from backend.app.services.retrieval import build_vectorstore

router = APIRouter()

@router.post("/upload/pdf")
async def upload_pdf(files: List[UploadFile] = File(...), session_id: str = Form("default")):
    """Upload PDF files, parse their structures, and upload embeddings to Pinecone."""
    ws = get_session(session_id)
    ws.is_processing = True
    ws.processing_message = f"Uploading and processing {len(files)} PDF documents..."
    
    try:
        all_docs = []
        new_names = []
        
        for file in files:
            file_path = os.path.join(DOWNLOADS_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Ingest PDF into split langchain documents
            file_docs = ingest_pdf(file_path, file.filename)
            all_docs.extend(file_docs)
            new_names.append(file.filename)
            
        if all_docs:
            ws.active_docs.extend(all_docs)
            ws.document_ids.extend(new_names)
            ws.active_image = None
            ws.active_image_name = None
            
            # Sync to Pinecone index
            vstore = build_vectorstore()
            vstore.add_documents(
                all_docs,
                ids=[f"{i}_{datetime.now().timestamp()}" for i in range(len(all_docs))]
            )
            
        ws.is_processing = False
        return {
            "status": "success",
            "message": f"Successfully indexed {len(all_docs)} chunks across {len(files)} PDFs!",
            "document_ids": ws.document_ids
        }
    except Exception as e:
        ws.is_processing = False
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/media")
async def upload_media(file: UploadFile = File(...), session_id: str = Form("default")):
    """Upload local audio/video media file, run Whisper transcription, and index text chunks."""
    ws = get_session(session_id)
    ws.is_processing = True
    ws.processing_message = f"Transcribing local media file '{file.filename}' (Whisper tiny running on CPU)..."
    
    try:
        media_path = os.path.join(DOWNLOADS_DIR, file.filename)
        with open(media_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Transcribe with Whisper model (runs in executor to prevent thread blocks)
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(None, lambda: whisper_model.transcribe(media_path))
        
        # Format text into chunk sizes suitable for index retrieval
        ts_chunks = create_timestamped_chunks(list(segments), chunk_size=500)
        file_docs = ingest_local_media(ts_chunks, file.filename, file.filename)
        
        if file_docs:
            ws.active_docs.extend(file_docs)
            ws.document_ids.append(file.filename)
            ws.active_image = None
            ws.active_image_name = None
            
            # Build vector embedding representation and write to DB
            vstore = build_vectorstore()
            vstore.add_documents(
                file_docs,
                ids=[f"{i}_{datetime.now().timestamp()}" for i in range(len(file_docs))]
            )
            
        ws.is_processing = False
        return {
            "status": "success",
            "message": f"Successfully indexed {len(file_docs)} timeline segments from local media!",
            "document_ids": ws.document_ids
        }
    except Exception as e:
        ws.is_processing = False
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...), session_id: str = Form("default")):
    """Upload visual image asset and mount as the active multimodal context."""
    ws = get_session(session_id)
    ws.is_processing = True
    ws.processing_message = "Mounting image asset..."
    
    try:
        image_bytes = await file.read()
        ws.active_image = image_bytes
        ws.active_image_name = file.filename
        ws.document_ids = [file.filename]
        ws.active_docs = []  # Clear active text docs when in image mode
        
        ws.is_processing = False
        return {
            "status": "success",
            "message": f"Successfully loaded image '{file.filename}' as knowledge base resource!",
            "document_ids": ws.document_ids
        }
    except Exception as e:
        ws.is_processing = False
        raise HTTPException(status_code=500, detail=str(e))
