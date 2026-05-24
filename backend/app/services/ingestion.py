import os
import requests
import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.config import SERPAPI_API_KEY
from backend.app.utils.helpers import clean_ocr_text, extract_yt_identifier

def ingest_pdf(pdf_path: str, filename: str) -> List[Document]:
    """Parse a PDF document using PyMuPDF and split it into text chunks."""
    document_parser = PyMuPDFLoader(pdf_path)  
    scraped_pages = document_parser.load()
    
    content_divider = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    divided_sections = content_divider.split_documents(scraped_pages)
    
    assembled_knowledge_nodes = []
    for segment_idx, text_block in enumerate(divided_sections):
        sanitized_text = clean_ocr_text(text_block.page_content)
        knowledge_doc = Document(
            page_content=sanitized_text, 
            metadata={
                "document_id": filename, 
                "source_type": "pdf", 
                "source_name": filename, 
                "page": text_block.metadata.get("page", -1), 
                "timestamp": "", 
                "chunk_id": segment_idx
            }
        )
        assembled_knowledge_nodes.append(knowledge_doc)
    return assembled_knowledge_nodes

def ingest_youtube(video_url: str, doc_reference_id: str) -> List[Document]:
    """Retrieve video transcripts from SerpApi and split into chunks."""
    target_vid_id = extract_yt_identifier(video_url)
    if not target_vid_id:
        return []
        
    # Fetch Metadata (Title, Channel)
    vid_info_config = {"api_key": SERPAPI_API_KEY, "engine": "youtube_video", "v": target_vid_id}
    try:
        metadata_payload = requests.get("https://serpapi.com/search", params=vid_info_config).json()
    except Exception:
        metadata_payload = {}
        
    vid_title = metadata_payload.get("title", f"Video {target_vid_id}")
    creator_channel = metadata_payload.get("channel", {}).get("name", "Unknown Channel")
    
    # Fetch Transcript Segments
    transcript_config = {
        "api_key": SERPAPI_API_KEY, 
        "engine": "youtube_video_transcript", 
        "v": target_vid_id, 
        "type": "asr"
    }
    try:
        spoken_payload = requests.get("https://serpapi.com/search", params=transcript_config).json()
    except Exception:
        spoken_payload = {}
    
    if "error" in spoken_payload or not spoken_payload.get("transcript"):
        return []
        
    # Chunk the transcript dynamically
    assembled_knowledge_nodes = []
    running_text_buffer = ""
    initial_timestamp = None
    segment_counter = 0
    
    for speech_block in spoken_payload.get("transcript", []):
        raw_spoken_line = speech_block.get("snippet", "").strip()
        if not raw_spoken_line: 
            continue
        
        if initial_timestamp is None: 
            initial_timestamp = speech_block.get("start_time_text")
            
        running_text_buffer += " " + raw_spoken_line
        
        # When chunk reaches ~500 chars, write it and reset buffer
        if len(running_text_buffer) >= 500:
            final_timestamp = speech_block.get("start_time_text")
            knowledge_doc = Document(
                page_content=clean_ocr_text(running_text_buffer.strip()), 
                metadata={
                    "document_id": doc_reference_id, 
                    "source_type": "youtube", 
                    "source_name": vid_title, 
                    "video_id": target_vid_id, 
                    "channel_name": creator_channel, 
                    "page": -1, 
                    "timestamp": f"{initial_timestamp} - {final_timestamp}", 
                    "chunk_id": segment_counter
                }
            )
            assembled_knowledge_nodes.append(knowledge_doc)
            segment_counter += 1
            running_text_buffer = ""
            initial_timestamp = None
            
    # Append any remaining transcript buffer
    if running_text_buffer:
        knowledge_doc = Document(
            page_content=clean_ocr_text(running_text_buffer.strip()), 
            metadata={
                "document_id": doc_reference_id, 
                "source_type": "youtube", 
                "source_name": vid_title, 
                "video_id": target_vid_id, 
                "channel_name": creator_channel, 
                "page": -1, 
                "timestamp": f"{initial_timestamp} - End", 
                "chunk_id": segment_counter
            }
        )
        assembled_knowledge_nodes.append(knowledge_doc)
        
    return assembled_knowledge_nodes

def ingest_local_media(temporal_segments, target_filename: str, doc_reference_id: str) -> List[Document]:
    """Map timestamped Whisper segments to LangChain document chunks."""
    assembled_knowledge_nodes = []
    for segment_idx, media_block in enumerate(temporal_segments):
        knowledge_doc = Document(
            page_content=media_block["text"], 
            metadata={
                "document_id": doc_reference_id, 
                "source_type": "media_file", 
                "source_name": target_filename, 
                "page": -1, 
                "timestamp": f'{media_block["start_time"]}s - {media_block["end_time"]}s', 
                "chunk_id": segment_idx
            }
        )
        assembled_knowledge_nodes.append(knowledge_doc)
    return assembled_knowledge_nodes

def ingest_website(url_link: str, doc_reference_id: str) -> List[Document]:
    """Scrape web pages, clean HTML structures and ingest split chunks."""
    fetch_parameters = {
        "header_template": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", 
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8", 
            "Accept-Language": "en-US,en;q=0.5"
        }
    }
    document_parser = AsyncHtmlLoader([url_link], **fetch_parameters)
    
    try: 
        unprocessed_web_data = asyncio.run(asyncio.wait_for(asyncio.to_thread(document_parser.load), timeout=15.0))
    except Exception:
        return []
        
    if not unprocessed_web_data or not unprocessed_web_data[0].page_content: 
        return []
        
    markup_cleaner = BeautifulSoupTransformer()
    filtered_web_data = markup_cleaner.transform_documents(
        unprocessed_web_data, 
        tags_to_extract=["p", "li", "h1", "h2", "h3", "div"]
    )
    
    content_divider = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    divided_sections = content_divider.split_documents(filtered_web_data)
    
    assembled_knowledge_nodes = []
    for segment_idx, text_block in enumerate(divided_sections):
        sanitized_text = clean_ocr_text(text_block.page_content)
        if len(sanitized_text) < 40: 
            continue
        knowledge_doc = Document(
            page_content=sanitized_text, 
            metadata={
                "document_id": doc_reference_id, 
                "source_type": "website", 
                "source_name": url_link.replace("https://", "").replace("www.", "").split("/")[0], 
                "page": -1, 
                "timestamp": "", 
                "chunk_id": segment_idx
            }
        )
        assembled_knowledge_nodes.append(knowledge_doc)
    return assembled_knowledge_nodes
