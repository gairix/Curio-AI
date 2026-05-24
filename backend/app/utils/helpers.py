import re
import base64
import urllib.parse
from typing import List, Optional
from langchain_core.documents import Document

def clean_ocr_text(text: str) -> str:
    """Fix broken camelCase and spacing around punctuation common in OCR or raw HTML."""
    if not text: 
        return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([,:;?])', r'\1\2 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def encode_bytes_to_base64(image_bytes: bytes) -> str:
    """Encode binary image payload to base64 string for multimodal LLM processing."""
    return base64.b64encode(image_bytes).decode("utf-8")

def format_docs(retrieved_docs: List[Document]) -> str:
    """Compile retrieved documents list into a formatted string for LLM context."""
    compiled_string = ""
    for doc_item in retrieved_docs:
        meta_details = doc_item.metadata
        compiled_string += f"\n\nSOURCE ({meta_details['source_type'].upper()}): {meta_details['source_name']}\nCONTENT:\n{doc_item.page_content}"
    return compiled_string

def extract_yt_identifier(video_link: str) -> Optional[str]:
    """Parse standard, shortened, shorts, mobile or embed YouTube URLs to extract the raw video ID."""
    if not video_link:
        return None
    # Regex to capture 11-char video ID from various YouTube URL patterns
    pattern = r'(?:https?://)?(?:www\.|m\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, video_link.strip())
    if match:
        return match.group(1)
    return None

def create_timestamped_chunks(segments, chunk_size: int = 500) -> List[dict]:
    """Group Whisper raw transcription segments into larger timestamp-anchored text blocks."""
    chunks, current_text, start_time, end_time = [], "", None, None
    for segment in segments:
        text = segment.text.strip()
        if start_time is None: 
            start_time = segment.start
        end_time = segment.end
        current_text += " " + text
        
        if len(current_text) >= chunk_size:
            chunks.append({
                "text": current_text.strip(), 
                "start_time": round(start_time, 2), 
                "end_time": round(end_time, 2)
            })
            current_text, start_time, end_time = "", None, None
            
    if current_text:
        chunks.append({
            "text": current_text.strip(), 
            "start_time": round(start_time, 2) if start_time is not None else 0.0, 
            "end_time": round(end_time, 2) if end_time is not None else 0.0
        })
    return chunks
