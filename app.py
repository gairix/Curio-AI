import os
import re
import base64
from datetime import datetime
import streamlit as st
import yt_dlp
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field
import urllib.parse
import requests
import asyncio

# LangChain & Vector Infrastructure
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# Chat Memory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer


st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS — MODERN GLASSMORPHIC AI UI


st.markdown("""
<style>

/* =========================================================
GLOBAL SETTINGS
========================================================= */

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(124,58,237,0.12), transparent 30%),
        radial-gradient(circle at top right, rgba(59,130,246,0.10), transparent 30%),
        linear-gradient(180deg, #050816 0%, #020617 100%);
    color: #ffffff;
}

/* =========================================================
MAIN CONTENT AREA
========================================================= */

.block-container {
    max-width: 1300px;
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 1rem;
}

/* =========================================================
SIDEBAR — LARGE PROFESSIONAL PANEL
========================================================= */

[data-testid="stSidebar"] {

    min-width: 390px !important;
    max-width: 390px !important;

    background:
        linear-gradient(
            180deg,
            #070B1A 0%,
            #020617 100%
        );

    border-right: 1px solid rgba(255,255,255,0.08);

    padding-top: 1rem;
}

[data-testid="stSidebarContent"] {
    padding: 1rem;
}

/* Sidebar Headers */

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {

    color: white !important;
    font-weight: 700;
    letter-spacing: 0.3px;
}

/* =========================================================
HERO TITLE
========================================================= */

.hero-title {

    font-size: 3.5rem;
    font-weight: 800;

    background: linear-gradient(
        90deg,
        #c084fc,
        #8b5cf6,
        #6366f1
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 0.2rem;
}

.hero-subtitle {

    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.3rem;
}

/* =========================================================
METRIC CARDS
========================================================= */

[data-testid="metric-container"] {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.04),
            rgba(255,255,255,0.02)
        );

    border: 1px solid rgba(255,255,255,0.06);

    padding: 1.2rem;

    border-radius: 22px;

    backdrop-filter: blur(16px);

    box-shadow:
        0px 6px 28px rgba(0,0,0,0.25);

    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover {

    transform: translateY(-3px);

    border: 1px solid rgba(139,92,246,0.35);
}

/* =========================================================
UPLOAD BOXES
========================================================= */

[data-testid="stFileUploader"] {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.03),
            rgba(255,255,255,0.01)
        );

    border: 1px dashed rgba(139,92,246,0.35);

    border-radius: 20px;

    padding: 1.3rem;

    backdrop-filter: blur(12px);
}

/* =========================================================
TEXT INPUTS
========================================================= */

.stTextInput input,
.stTextArea textarea {

    background: rgba(255,255,255,0.04) !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    color: white !important;

    border-radius: 18px !important;

    padding: 1rem !important;

    font-size: 0.96rem !important;

    backdrop-filter: blur(10px);
}

/* =========================================================
BUTTONS
========================================================= */

.stButton > button {

    width: 100%;

    border-radius: 16px;

    border: none;

    padding: 0.9rem;

    font-weight: 700;

    color: white;

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #6366f1
        );

    transition: all 0.3s ease;

    box-shadow:
        0px 6px 22px rgba(124,58,237,0.25);
}

.stButton > button:hover {

    transform: translateY(-2px);

    box-shadow:
        0px 10px 30px rgba(124,58,237,0.40);
}

/* =========================================================
CHAT SECTION
========================================================= */

[data-testid="stChatMessage"] {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.03),
            rgba(255,255,255,0.015)
        );

    border: 1px solid rgba(255,255,255,0.06);

    border-radius: 24px;

    padding: 1.4rem;

    margin-bottom: 1.2rem;

    backdrop-filter: blur(16px);

    box-shadow:
        0px 6px 30px rgba(0,0,0,0.25);

    transition: all 0.3s ease;
}

[data-testid="stChatMessage"]:hover {

    border: 1px solid rgba(139,92,246,0.18);

    transform: translateY(-2px);
}

/* Assistant Messages */

[data-testid="stChatMessage"]:has([aria-label="assistant"]) {

    background:
        linear-gradient(
            135deg,
            rgba(124,58,237,0.08),
            rgba(59,130,246,0.05)
        );
}

/* User Messages */

[data-testid="stChatMessage"]:has([aria-label="user"]) {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.04),
            rgba(255,255,255,0.02)
        );
}

/* =========================================================
CHAT INPUT AREA
========================================================= */

[data-testid="stChatInput"] {

    background:
        rgba(2,6,23,0.94);

    border-top:
        1px solid rgba(255,255,255,0.08);

    padding-top: 1rem;
}

[data-testid="stChatInput"] textarea {

    background:
        rgba(255,255,255,0.04) !important;

    border:
        1px solid rgba(255,255,255,0.08) !important;

    border-radius: 22px !important;

    color: white !important;

    padding: 1rem !important;

    font-size: 1rem !important;

    backdrop-filter: blur(10px);
}

/* =========================================================
SOURCE CARDS
========================================================= */

.source-card {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.03),
            rgba(255,255,255,0.015)
        );

    border:
        1px solid rgba(255,255,255,0.06);

    border-radius: 18px;

    padding: 1rem;

    margin-bottom: 0.7rem;

    transition: all 0.3s ease;
}

.source-card:hover {

    border:
        1px solid rgba(139,92,246,0.25);

    transform: translateY(-2px);
}

/* =========================================================
DIVIDERS
========================================================= */

hr {

    border-color:
        rgba(255,255,255,0.08);
}

/* =========================================================
SCROLLBAR
========================================================= */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {

    background:
        rgba(255,255,255,0.15);

    border-radius: 20px;
}

/* =========================================================
HIDE STREAMLIT BRANDING
========================================================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* =========================================================
TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {

    gap: 10px;
}

.stTabs [data-baseweb="tab"] {

    background:
        rgba(255,255,255,0.03);

    border-radius: 14px;

    padding:
        10px 18px;

    color:
        white;

    font-weight:
        600;

    border:
        1px solid rgba(255,255,255,0.05);
}

.stTabs [aria-selected="true"] {

    background:
        linear-gradient(
            135deg,
            #7c3aed,
            #6366f1
        ) !important;

    color:
        white !important;
}

/* =========================================================
INFO BOXES
========================================================= */

.stAlert {

    border-radius: 16px;
}

/* =========================================================
SUGGESTIONS WRAPPER PANEL
========================================================= */
.suggestion-btn-container {
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}

</style>
""", unsafe_allow_html=True)


# Global state setup for ongoing Streamlit refreshes
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "active_docs" not in st.session_state:
    st.session_state.active_docs = []
if "store" not in st.session_state:
    st.session_state.store = {}
if "active_image" not in st.session_state:
    st.session_state.active_image = None
    
# State counters to force-reset widget states instantly
if "pdf_uploader_key" not in st.session_state:
    st.session_state.pdf_uploader_key = 0
if "yt_input_key" not in st.session_state:
    st.session_state.yt_input_key = 0
if "media_uploader_key" not in st.session_state:
    st.session_state.media_uploader_key = 0
if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0

# Add these along with your other session state definitions at the top
if "active_quiz_data" not in st.session_state:
    st.session_state.active_quiz_data = None
if "active_summary_data" not in st.session_state:
    st.session_state.active_summary_data = None
if "active_comparison_data" not in st.session_state:
    st.session_state.active_comparison_data = None


# CONFIGURATION & ENVIRONMENT SETUP

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
serpapi_access_token = os.getenv("SERPAPI_API_KEY")


# ==========================================
# SECTION 2: ML MODEL CACHING & INITIALIZATION
# ==========================================
# Loads heavy Machine Learning models once and caches them to prevent slow reloads 
# every time a new request is made to the server.

@st.cache_resource
def boot_core_ml_engines():
    text_vectorizer = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": "cpu"}, 
        encode_kwargs={"normalize_embeddings": True}
    )
    relevance_scorer = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    
    # ADDED: Boot up the Whisper model for local files
    whisper_engine = WhisperModel("tiny", device="cpu", compute_type="float32")
    
    return text_vectorizer, relevance_scorer, whisper_engine

# Unpack all three engines
text_vectorizer, relevance_scorer, whisper_model = boot_core_ml_engines()


# ==========================================
# SECTION 3: TEXT CLEANING UTILITIES
# ==========================================
# Helper functions to clean up messy text commonly retrieved from OCR, PDFs, or raw HTML.
def clean_ocr_text(text):
    if not text: 
        return ""
    # Fix broken camelCase and spacing around punctuation
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([,:;?])', r'\1\2 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ==========================================
# SECTION 4: DATA INGESTION PIPELINES
# ==========================================
# Distinct functions to ingest, chunk, and format different types of media into unified Document objects.

# --- 4A: PDF INGESTION ---
def ingest_pdf(pdf_path, document_id):
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
                "document_id": document_id, 
                "source_type": "pdf", 
                "source_name": os.path.basename(pdf_path), 
                "page": text_block.metadata.get("page", -1), 
                "timestamp": "", 
                "chunk_id": segment_idx
            }
        )
        assembled_knowledge_nodes.append(knowledge_doc)
    return assembled_knowledge_nodes

# --- 4B: YOUTUBE INGESTION (SERPAPI METHOD) ---
def extract_yt_identifier(video_link):
    url_components = urllib.parse.urlparse(video_link)
    if url_components.hostname == 'youtu.be': 
        return url_components.path[1:]
    if url_components.hostname in ('www.youtube.com', 'youtube.com'):
        if url_components.path == '/watch': 
            return urllib.parse.parse_qs(url_components.query)['v'][0]
    return None

def ingest_youtube(video_url, doc_reference_id):
    target_vid_id = extract_yt_identifier(video_url)
    if not target_vid_id:
        st.error("Invalid YouTube URL provided.")
        return []
        
    # 1. Fetch Metadata (Title, Channel)
    vid_info_config = {"api_key": serpapi_access_token, "engine": "youtube_video", "v": target_vid_id}
    metadata_payload = requests.get("https://serpapi.com/search", params=vid_info_config).json()
    vid_title = metadata_payload.get("title", f"Video {target_vid_id}")
    creator_channel = metadata_payload.get("channel", {}).get("name", "Unknown Channel")
    
    # 2. Fetch Transcript Segments
    transcript_config = {"api_key": serpapi_access_token, "engine": "youtube_video_transcript", "v": target_vid_id, "type": "asr", "language_code": "en"}
    spoken_payload = requests.get("https://serpapi.com/search", params=transcript_config).json()
    
    if "error" in spoken_payload or not spoken_payload.get("transcript"):
        st.error(f"Sorry, no transcript available for '{vid_title}'.")
        return []
        
    # 3. Chunk the transcript dynamically
    assembled_knowledge_nodes = []
    running_text_buffer = ""
    initial_timestamp = None
    segment_counter = 0
    
    for speech_block in spoken_payload.get("transcript", []):
        raw_spoken_line = speech_block.get("snippet", "").strip()
        if not raw_spoken_line: continue
        
        if initial_timestamp is None: 
            initial_timestamp = speech_block.get("start_time_text")
            
        running_text_buffer += " " + raw_spoken_line
        
        # Once chunk reaches ~500 chars, save it and reset buffer
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
            
    # Append any remaining text as the final chunk
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

# --- 4C: WEBSITE URL INGESTION ---
def ingest_website(url_link, doc_reference_id):
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
    except asyncio.TimeoutError:
        st.error(f"⏱️ Connection Timed Out: '{url_link}' took too long to respond.")
        return []
    except Exception as err:
        st.error(f"⚠️ Failed connecting to web asset framework: {err}")
        return []
        
    if not unprocessed_web_data or not unprocessed_web_data[0].page_content: 
        return []
        
    markup_cleaner = BeautifulSoupTransformer()
    filtered_web_data = markup_cleaner.transform_documents(unprocessed_web_data, tags_to_extract=["p", "li", "h1", "h2", "h3", "div"])
    
    content_divider = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    divided_sections = content_divider.split_documents(filtered_web_data)
    
    assembled_knowledge_nodes = []
    for segment_idx, text_block in enumerate(divided_sections):
        sanitized_text = clean_ocr_text(text_block.page_content)
        if len(sanitized_text) < 40: continue
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

# ADDED: Helper function to chunk Whisper's raw audio segments based on text length
def create_timestamped_chunks(segments, chunk_size=500):
    chunks, current_text, start_time, end_time = [], "", None, None
    for segment in segments:
        text = segment.text.strip()
        if start_time is None: 
            start_time = segment.start
        end_time = segment.end
        current_text += " " + text
        
        if len(current_text) >= chunk_size:
            chunks.append({"text": current_text.strip(), "start_time": round(start_time, 2), "end_time": round(end_time, 2)})
            current_text, start_time, end_time = "", None, None
            
    if current_text:
        chunks.append({"text": current_text.strip(), "start_time": round(start_time, 2), "end_time": round(end_time, 2)})
    return chunks

# --- 4D: LOCAL MEDIA INGESTION ---
def ingest_local_media(temporal_segments, target_filename, doc_reference_id):
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


# ==========================================
# SECTION 5: VECTOR DATABASE SETUP
# ==========================================
# Connects to Pinecone and initializes the cloud vector store index.
def build_vectorstore():
    try: 
        pc_instance = Pinecone(api_key=pinecone_api_key)
    except Exception as err:
        st.error(f"Pinecone initialization failed: {err}")
        st.stop()
        
    db_namespace = "ai-learning-assistant"
    active_db_response = pc_instance.list_indexes()
    active_db_list = (active_db_response.names() if hasattr(active_db_response, "names") else [idx_obj.name for idx_obj in active_db_response])
    
    if db_namespace not in active_db_list: 
        pc_instance.create_index(
            name=db_namespace, 
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
    return PineconeVectorStore(index_name=db_namespace, embedding=text_vectorizer, pinecone_api_key=pinecone_api_key)


# ==========================================
# SECTION 6: LLM & OUTPUT PARSER CONFIGURATION
# ==========================================
# Defines the specific data schemas we expect back from the LLM, initializes 
# the Llama 3 models via Groq, and sets up prompts.

# Enforced Pydantic Schemas
class RAGResponse(BaseModel): 
    answer: str = Field(description="Answer generated from context")

class QuizQuestion(BaseModel):
    question: str = Field(description="The multiple choice question prompt text")
    options: list[str] = Field(description="Exactly 4 clean string answers options to choose from")
    correct_answer: str = Field(description="The exact match string identical to the correct option choice")
    explanation: str = Field(description="High clarity overview explaining why this selection is accurate")

class QuizSchema(BaseModel): 
    quiz: list[QuizQuestion] = Field(description="An explicit collection of 3-5 high analytical study questions")

parser = PydanticOutputParser(pydantic_object=RAGResponse)
quiz_parser = PydanticOutputParser(pydantic_object=QuizSchema)

# Model Initialization
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, groq_api_key=groq_api_key)

def format_docs(retrieved_docs):
    compiled_string = ""
    for doc_item in retrieved_docs:
        meta_details = doc_item.metadata
        compiled_string += f"\n\nSOURCE ({meta_details['source_type'].upper()}): {meta_details['source_name']}\nCONTENT:\n{doc_item.page_content}"
    return compiled_string

# Base Prompts (Truncated in visual structure, keep string templates intact)
pdf_prompt = ChatPromptTemplate.from_template("You are an expert educational AI learning assistant helping users learn from complex PDFs. Analyze the provided context carefully and answer the user's question with deep conceptual clarity, structural elegance, and professional formatting. CRITICAL PARSING RULE: - You MUST respond ONLY with a valid JSON object matching the requested schema. - Do NOT include any introduction conversational text, headers (like ###), bullet points, or markdown text outside of the JSON block itself. - Your entire response text must wrap directly into the raw JSON properties required below. IMPORTANT RESPONSE RULES: - Use clean Markdown layout elements: Utilize clear headers (###), bold key phrases, and break explanations down into bullet points or numbered lists. No walls of prose. - Connect ideas and synthesize data gracefully even if the user asks an indirect question. - Rely ONLY on the provided context. Do not hallucinate external legal or historical facts. - If the context completely lacks any relevant information, state exactly: 'I could not find enough matching information in the provided PDF to answer that safely.' CONTEXT: {context} CHAT HISTORY: {history} QUESTION: {question} {format_instructions}")

youtube_prompt = ChatPromptTemplate.from_template("You are an expert AI assistant helping users master knowledge from audio lectures and video clips. Analyze the provided transcript carefully and answer the user's question with high structural organization and professional formatting. CRITICAL PARSING RULE: - You MUST respond ONLY with a valid JSON object matching the requested schema. - Do NOT include any introduction conversational text, headers (like ###), bullet points, or markdown text outside of the JSON block itself. - Your entire response text must wrap directly into the raw JSON properties required below. IMPORTANT RESPONSE RULES: - Use clean Markdown layout elements: Utilize clear headers (###), bold key phrases, and break explanations down into bullet points or numbered lists. No walls of prose. - Synthesize ideas fluidly across timestamps to create an insightful, clear, and comprehensive answer. - Rely ONLY on the provided context. Do not hallucinate outside details. - If the context completely lacks any relevant information, state exactly: 'I could not find enough matching information in the lecture to answer that safely.' CONTEXT: {context} CHAT HISTORY: {history} QUESTION: {question} {format_instructions}")

contextualize_q_prompt = ChatPromptTemplate.from_template("Given the conversation history and the latest user question, rewrite it into a standalone question. Do NOT answer it. CHAT HISTORY: {history}\nQUESTION: {question}\nStandalone Question:")


# ==========================================
# SECTION 7: RAG CHAINS & SESSION MEMORY
# ==========================================
# Connects the prompts, parsers, LLM, and session state to run queries against history.

pdf_chain = ({"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"], "format_instructions": lambda _: parser.get_format_instructions(), "history": lambda x: x.get("history", "")} | pdf_prompt | llm | parser)
youtube_chain = ({"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"], "format_instructions": lambda _: parser.get_format_instructions(), "history": lambda x: x.get("history", "")} | youtube_prompt | llm | parser)

rag_branch = RunnableBranch(
    (lambda x: len(x["docs"]) > 0 and x["docs"][0].metadata.get("source_type") == "pdf", pdf_chain), 
    (lambda x: len(x["docs"]) > 0 and x["docs"][0].metadata.get("source_type") == "youtube", youtube_chain), 
    pdf_chain
)

question_rewriter = contextualize_q_prompt | llm

def get_session_history(session_tracker_id):
    if session_tracker_id not in st.session_state.store: 
        st.session_state.store[session_tracker_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_tracker_id]

conversational_rag = RunnableWithMessageHistory(rag_branch, get_session_history, input_messages_key="question", history_messages_key="history")

def get_contextualized_question(user_query, session_tracker_id="user_1"):
    chat_log = get_session_history(session_tracker_id)
    chat_log_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_log.messages])
    return question_rewriter.invoke({"history": chat_log_text, "question": user_query}).content


# ==========================================
# SECTION 8: HYBRID RETRIEVAL ENGINE
# ==========================================
# Queries both standard keyword search (BM25) and Semantic Vector search (Pinecone), 
# then re-ranks the combined pool to fetch the absolute best context for the LLM.
def execute_hybrid_retrieval(user_query, db_vectorstore, doc_reference_id):
    if not st.session_state.active_docs: return []
    
    # 1. Lexical Retrieval (Keyword Matching)
    lexical_retriever = BM25Retriever.from_documents(st.session_state.active_docs)
    lexical_retriever.k = 6
    
    # 2. Semantic Retrieval (Vector Similarity)
    search_criteria = {"$or": [{"source_name": {"$in": doc_reference_id}}, {"video_id": {"$in": doc_reference_id}}]} if isinstance(doc_reference_id, list) else {"document_id": doc_reference_id}
    semantic_retriever = db_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.5, "filter": search_criteria})
    
    # Fire off both queries
    lexical_results = lexical_retriever.invoke(user_query)
    semantic_results = semantic_retriever.invoke(user_query)
    
    # 3. Deduplicate Combined Results
    tracked_strings, merged_results = set(), []
    for knowledge_doc in (lexical_results + semantic_results):
        if knowledge_doc.page_content.strip() not in tracked_strings:
            tracked_strings.add(knowledge_doc.page_content.strip())
            merged_results.append(knowledge_doc)
            
    if not merged_results: return []
    
    # 4. Cross-Encoder Re-ranking
    eval_pairs = [(user_query, knowledge_doc.page_content) for knowledge_doc in merged_results]
    scoring_weights = relevance_scorer.predict(eval_pairs)
    
    # Sort by the highest relevance scores and return top 4
    sorted_eval = sorted(zip(merged_results, scoring_weights), key=lambda x: x[1], reverse=True)
    return [knowledge_doc for knowledge_doc, weight in sorted_eval[:4]]


# HERO HEADER


st.markdown("""
<div style="text-align:center; padding-top:1rem; padding-bottom:1rem;">

<div class="hero-title">
🧠 AI Learning Assistant
</div>

<div class="hero-subtitle">
Advanced Multi-Source Retrieval-Augmented Generation System for Knowledge Processing
</div>

</div>
""", unsafe_allow_html=True)


# SIDEBAR CONFIGURATIONS

with st.sidebar:

    st.markdown("## ⚙️ Workspace")

    tab2, tab1, tab4, tab3 = st.tabs(["▶️ URL","📄 PDFs", "📸 Image", "🎙️ Audio/Video"])

    # PDF TAB
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_uploader_{st.session_state.pdf_uploader_key}"
        )

        if uploaded_files and st.button("🚀 Process PDFs"):
            progress = st.progress(0)
            with st.spinner("Processing PDF resources..."):
                all_docs = []
                os.makedirs("downloads", exist_ok=True)
                progress.progress(10, text="Saving uploaded assets...")

                for uploaded_file in uploaded_files:
                    local_path = os.path.join("downloads", uploaded_file.name)
                    with open(local_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    progress.progress(35, text=f"Analyzing {uploaded_file.name}...")
                    file_docs = ingest_pdf(local_path, uploaded_file.name)
                    all_docs.extend(file_docs)

                progress.progress(65, text="Building vector index...")
                st.session_state.active_docs = all_docs
                st.session_state.document_id = [f.name for f in uploaded_files]
                st.session_state.active_image = None  

                vstore = build_vectorstore()
                vstore.add_documents(
                    all_docs,
                    ids=[f"{i}_{datetime.now().timestamp()}" for i in range(len(all_docs))]
                )
                progress.progress(100, text="Completed!")
                st.success(f"Successfully indexed {len(all_docs)} chunks across {len(uploaded_files)} PDFs!")

    # 4. UNIVERSAL URL TAB (REPLACES EXCLUSIVE YT LOGIC)
    with tab2:
        urls_input = st.text_area(
            "Paste YouTube Links (comma separated)",
            key=f"yt_input_{st.session_state.yt_input_key}"
        )

        if urls_input and st.button("🌐 Process YouTube Content"):
            progress = st.progress(0)
            
            urls = [url.strip() for url in urls_input.split(",") if url.strip()]
            all_processed_docs = []
            source_ids_list = []
            total = len(urls)

            for idx, target_url in enumerate(urls):
                try:
                    progress.progress(int((idx / total) * 100), text=f"Processing Link {idx+1}/{total}")
                    
                    # Enforce YouTube-only logic
                    if "youtube.com" in target_url or "youtu.be" in target_url:
                        with st.spinner(f"Extracting SerpApi transcript for: {target_url}..."):
                            
                            # Grab the video ID to use as our document reference
                            target_vid_id = extract_yt_identifier(target_url)
                            
                            if not target_vid_id:
                                st.error(f"Could not extract Video ID from {target_url}")
                                continue
                                
                            # Feed directly into the new SerpApi ingestion function
                            file_docs = ingest_youtube(target_url, target_vid_id)
                            
                            # Only add to the vector store payload if SerpApi successfully found a transcript
                            if file_docs:
                                source_ids_list.append(target_vid_id)
                                all_processed_docs.extend(file_docs)
                    else:
                        st.warning(f"Skipping non-YouTube link: {target_url}")

                except Exception as e:
                    st.error(f"Failed processing YouTube link: {target_url}")
                    st.exception(e)

            # Update Session State and Vector Store
            if all_processed_docs:
                st.session_state.active_docs = all_processed_docs
                st.session_state.document_id = source_ids_list
                st.session_state.active_image = None  

                vstore = build_vectorstore()
                vstore.add_documents(
                    all_processed_docs,
                    ids=[f"{i}_{datetime.now().timestamp()}" for i in range(len(all_processed_docs))]
                )
                progress.progress(100, text="Completed!")
                st.success(f"Indexed {len(all_processed_docs)} granular context segments across {len(source_ids_list)} YouTube videos!")

    # AUDIO/VIDEO UPLOADER TAB
    with tab3:
        uploaded_media = st.file_uploader(
            "Upload Local Audio/Video Recording",
            type=["mp3", "wav", "m4a", "mp4"],
            key=f"media_uploader_{st.session_state.media_uploader_key}"
        )
        
        if uploaded_media and st.button("⚡ Process Local Media File"):
            progress = st.progress(0)
            with st.spinner("Transcribing local audio binary streams..."):
                os.makedirs("downloads", exist_ok=True)
                media_path = os.path.join("downloads", uploaded_media.name)
                
                with open(media_path, "wb") as f:
                    f.write(uploaded_media.getbuffer())
                    
                progress.progress(30, text="Initializing model transcription layers...")
                segments, _ = whisper_model.transcribe(media_path)
                
                progress.progress(60, text="Configuring granular milestone chunks...")
                ts_chunks = create_timestamped_chunks(list(segments), chunk_size=500)
                file_docs = ingest_local_media(ts_chunks, uploaded_media.name, uploaded_media.name)
                
                progress.progress(85, text="Injecting timeline configurations to vector store...")
                st.session_state.active_docs = file_docs
                st.session_state.document_id = [uploaded_media.name]
                st.session_state.active_image = None
                
                vstore = build_vectorstore()
                vstore.add_documents(
                    file_docs,
                    ids=[f"{i}_{datetime.now().timestamp()}" for i in range(len(file_docs))]
                )
                progress.progress(100, text="Completed!")
                st.success(f"Successfully processed and indexed {len(file_docs)} timeline blocks from media asset!")

    # IMAGE TAB
    with tab4:
        uploaded_image = st.file_uploader(
            "Upload Image Asset",
            type=["png", "jpg", "jpeg"],
            key=f"image_uploader_{st.session_state.image_uploader_key}"
        )

        if uploaded_image and st.button("👁️ Process Image"):
            with st.spinner("Analyzing image asset structures..."):
                st.session_state.active_image = uploaded_image.getvalue()
                st.session_state.document_id = [uploaded_image.name]
                st.session_state.active_docs = [] 
                
                st.image(st.session_state.active_image, caption="Active Image Preview", width=280)
                st.success(f"Successfully mounted '{uploaded_image.name}' as visual knowledge base resource!")

    # ACTIVE SOURCES DISPLAY
    st.markdown("---")
    st.markdown("## 📚 Active Knowledge Base")
    if st.session_state.document_id:
        for doc in st.session_state.document_id:
            st.success(doc)
    else:
        st.info("No active resources loaded.")

    # RESET CONTROLS
    st.markdown("---")
    st.markdown("## 🧹 Session Controls")
    if st.button("💬 Clear Chat History"):
        st.session_state.messages = []
        if "user_1" in st.session_state.store:
            del st.session_state.store["user_1"]
        st.success("Chat history cleared!")
        st.rerun()

    if st.button("🗑️ Reset Entire Workspace"):
        st.session_state.messages = []
        st.session_state.document_id = None
        st.session_state.active_docs = []
        st.session_state.active_image = None

        st.session_state.active_quiz_data = None
        st.session_state.active_summary_data = None
        st.session_state.active_comparison_data = None

        if "user_1" in st.session_state.store:
            del st.session_state.store["user_1"]

        st.session_state.pdf_uploader_key += 1
        st.session_state.yt_input_key += 1
        st.session_state.media_uploader_key += 1
        st.session_state.image_uploader_key += 1
        st.success("Workspace reset complete!")
        st.rerun()


# =========================================================
# CHATSPACE ENVIRONMENT
# =========================================================

# =========================================================
# SUGGESTION INTERACTIONS WITH MULTIMODAL VISION FIX
# =========================================================
if st.session_state.document_id:
    # 1. Determine if we are in Image Mode or Text Mode
    is_image_mode = st.session_state.get("active_image") is not None

    if not is_image_mode:
        if st.session_state.active_docs:
            context_snippet = "\n".join([f"SOURCE: {d.metadata['source_name']} -> {d.page_content}" for d in st.session_state.active_docs[:12]])
        else:
            context_snippet = "No active text resources loaded."
    else:
        # Encode image to base64 so vision_llm can read it directly
        base64_repr = encode_bytes_to_base64(st.session_state.active_image)

    # Setup the 3 suggestion button columns
    suggest_col1, suggest_col2, suggest_col3 = st.columns(3)
    
    with suggest_col1:
        if st.button("✨ AI Summary", help="Generate a high-density executive summary card"):
            with st.spinner("Extracting conceptual themes and summarizing material..."):
                if not is_image_mode:
                    # Text Mode
                    summary_prompt = f"You are an elite academic compiler. Synthesize a clean, professional, high-density structured core study summary of the active resource material provided below. Utilize clean markdown headers (###), bold metrics, and succinct bullet points.\n\nMATERIAL CONTEXT:\n{context_snippet}"
                    st.session_state.active_summary_data = llm.invoke(summary_prompt).content
                else:
                    # 🚀 Vision Fix: Send the actual image bytes to the vision model
                    vision_payload = [
                        {"type": "text", "text": "You are an elite academic compiler. Closely analyze this image file and synthesize a clean, comprehensive, high-density study summary explaining all concepts, questions, or problems visible within it. Use clear headers (###), bold key terms, and explicit bullet points."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_repr}"}}
                    ]
                    st.session_state.active_summary_data = vision_llm.invoke([HumanMessage(content=vision_payload)]).content
                
                st.session_state.active_quiz_data = None
                st.session_state.active_comparison_data = None

    with suggest_col2:
        if st.button("🧩 Generate Quiz", help="Auto-generate an interactive analytical study evaluation"):
            with st.spinner("Synthesizing problem sets based on materials..."):
                if not is_image_mode:
                    # Text Mode
                    quiz_prompt = f"You are a university professor. Generate 3 high-quality multiple choice questions testing critical concepts explicitly covered in the context below. You MUST respond strictly with a valid JSON format adhering to the following instructions:\n{quiz_parser.get_format_instructions()}\n\nMATERIAL CONTEXT:\n{context_snippet}"
                    raw_quiz = llm.invoke(quiz_prompt)
                    st.session_state.active_quiz_data = quiz_parser.parse(raw_quiz.content)
                else:
                    # 🚀 Vision Fix: Let the vision model build a quiz directly from the image content
                    quiz_prompt = f"You are a university professor. Analyze this image and generate 3 high-quality multiple choice questions evaluating the concepts or solving the type of problem sets visible in the asset. You MUST respond strictly with a valid JSON format adhering to the following instructions:\n{quiz_parser.get_format_instructions()}"
                    vision_payload = [
                        {"type": "text", "text": quiz_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_repr}"}}
                    ]
                    raw_quiz = vision_llm.invoke([HumanMessage(content=vision_payload)])
                    st.session_state.active_quiz_data = quiz_parser.parse(raw_quiz.content)
                
                st.session_state.active_summary_data = None
                st.session_state.active_comparison_data = None

    with suggest_col3:
        if st.button("📊 Compare Assets", help="Execute an analytical cross-comparison matrix evaluation"):
            if is_image_mode:
                st.warning("⚠️ Cross-document comparison matrix modules are optimized for textual datasets and multiple loaded documents.")
            else:
                if len(st.session_state.document_id) < 2:
                    st.warning("⚠️ Comparative evaluations yield optimal metrics when multiple files are loaded together in the workspace.")
                with st.spinner("Executing context cross-matching and rendering structured matrix arrays..."):
                    matrix_prompt = f"You are an expert technical auditor. Provide a comprehensive cross-comparison matrix of the core arguments and insights across the loaded files. You MUST present your comparison inside a clean, comprehensive Markdown evaluation grid table with detailed column fields mapping similarities, distinct gaps, and conceptual differences.\n\nMATERIAL CONTEXTS FROM LOADED ACTIVE FILES:\n{context_snippet}"
                    st.session_state.active_comparison_data = llm.invoke(matrix_prompt).content
                    st.session_state.active_quiz_data = None
                    st.session_state.active_summary_data = None

    # -----------------------------------------------------------------
    # RENDER PERSISTENT PANELS FROM STATE
    # -----------------------------------------------------------------
    if st.session_state.active_summary_data:
        with st.container(border=True):
            st.markdown("### 📝 High-Density Executive Summary")
            st.markdown(st.session_state.active_summary_data)

    if st.session_state.active_comparison_data:
        with st.container(border=True):
            st.markdown("### 📊 Multi-Asset Matrix Comparison")
            st.markdown(st.session_state.active_comparison_data)

    if st.session_state.active_quiz_data:
        with st.container(border=True):
            st.markdown("### 🧩 Custom Analytical Evaluation Quiz")
            for idx, q in enumerate(st.session_state.active_quiz_data.quiz):
                st.markdown(f"**Q{idx+1}: {q.question}**")
                user_sel = st.radio("Choose Option:", q.options, index=None, key=f"suggest_state_q_{idx}")
                
                if user_sel is not None:
                    with st.expander("👁️ Check Answer & Explanation", expanded=True):
                        if user_sel == q.correct_answer: 
                            st.success(f"Correct! -> Answer: {q.correct_answer}")
                        else: 
                            st.error(f"Incorrect Selection. Correct option is: {q.correct_answer}")
                        st.caption(f"**Explanation:** {q.explanation}")
                st.markdown("<br>", unsafe_allow_html=True)

else:
    st.info("💡 Upload resource documents or assets in the workspace sidebar to unlock diagnostic study summaries, custom quizzes, and cross-comparators directly over the chat area.")


# CHAT HISTORY
for msg in st.session_state.messages:
    avatar = "👨‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            with st.container(border=True):
                st.markdown(msg["content"])

        if "references" in msg and msg["references"]:
            st.markdown("---")
            st.caption("📂 Verified Sources")
            for ref in msg["references"]:
                st.markdown(ref)


# CHAT INPUT
user_query = st.chat_input("Ask anything about your uploaded learning resources...")

if user_query:
    if not st.session_state.document_id:
        st.error("⚠️ Please upload and process resources before asking questions.")
    else:
        with st.chat_message("user", avatar="👨‍🎓"):
            st.markdown(user_query)

        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant", avatar="🤖"):
            # Multimodal Image check
            if st.session_state.get("active_image") is not None:
                with st.spinner("🧠 Analyzing image structure and synthesizing response..."):
                    base64_repr = encode_bytes_to_base64(st.session_state.active_image)
                    system_guideline = (
                        "You are an expert AI assistant specialized in analyzing visual diagrams, charts, and photos. "
                        "Answer with deep clarity, using headers (###), bold items, and lists. "
                        "Rely strictly on visual patterns observed in the image layout details."
                    )
                    content_payload = [
                        {"type": "text", "text": f"{system_guideline}\n\nUser Question: {user_query}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_repr}"}}
                    ]
                    vision_msg = HumanMessage(content=content_payload)
                    raw_vision_response = vision_llm.invoke([vision_msg])
                    answer_text = raw_vision_response.content
                    refs = [f"📸 Visual Context: {st.session_state.document_id[0]}"]

                with st.container(border=True):
                    st.markdown(answer_text)
                if refs:
                    st.markdown("---")
                    st.caption("📂 Verified Sources")
                    for ref in refs: st.markdown(ref)

            # Standard PDF/YouTube text cores check
            else:
                with st.status("🔍 Running Hybrid Retrieval Pipeline...", expanded=False) as status:
                    st.write("🔄 Contextualizing conversational query...")
                    standalone = get_contextualized_question(user_query)

                    st.write("⚡ Executing BM25 + Dense Retrieval...")
                    vstore = build_vectorstore()
                    retrieved_docs = execute_hybrid_retrieval(standalone, vstore, st.session_state.document_id)

                    st.write("📊 Applying Cross-Encoder Reranking...")
                    format_insts = parser.get_format_instructions()
                    chain_input = {"question": user_query, "docs": retrieved_docs, "history": "", "format_instructions": format_insts}
                    status.update(label="✅ Retrieval Complete", state="complete", expanded=False)

                with st.spinner("🧠 Synthesizing intelligent response..."):
                    response = conversational_rag.invoke(chain_input, config={"configurable": {"session_id": "user_1"}})
                    answer_text = response.answer if hasattr(response, 'answer') else str(response)
                    refs = []

                    for doc in retrieved_docs:
                        m = doc.metadata
                        if m['source_type'] == 'pdf':
                            clean_ref = f"📄 {m['source_name']} (Page {int(m['page']) + 1})"
                        else:
                            raw_timestamp = str(m.get("timestamp", "0"))
                            match = re.search(r"\d+(?:\.\d+)?", raw_timestamp)
                            start_seconds = int(float(match.group())) if match else 0
                            video_id = m.get('video_id', '')
                            clickable_url = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
                            clean_ref = f"🎥 {m['source_name']} [({m['timestamp']})]({clickable_url})"
                        if clean_ref not in refs: refs.append(clean_ref)

                    with st.container(border=True):
                        st.markdown(answer_text)
                    if refs:
                        st.markdown("---")
                        st.caption("📂 Verified Sources")
                        for ref in refs: st.markdown(ref)

        st.session_state.messages.append({"role": "assistant", "content": answer_text, "references": refs})