import os
import re
import base64
from datetime import datetime
import streamlit as st
import yt_dlp
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field

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
    max-width: 1600px;
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
ACTIVE KNOWLEDGE BASE BOX
========================================================= */

.knowledge-box {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.03),
            rgba(255,255,255,0.01)
        );

    border:
        1px solid rgba(255,255,255,0.05);

    border-radius:
        18px;

    padding:
        1rem;

    margin-top:
        0.5rem;
}

/* =========================================================
FOOTER
========================================================= */

.footer-text {

    text-align: center;

    color: #64748b;

    font-size: 0.9rem;

    margin-top: 1rem;
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
if "image_uploader_key" not in st.session_state:
    st.session_state.image_uploader_key = 0


# CONFIGURATION & ENVIRONMENT SETUP

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Cache heavy ML models so they only load ONCE when starting the server
@st.cache_resource
def load_ml_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fixed to 384-dim to match index safety
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    whisper = WhisperModel("tiny", device="cpu", compute_type="float32")
    return embeddings, reranker, whisper

embedding_model, rerank_model, whisper_model = load_ml_models()


# STEP 2 — PROCESSING LOGIC WITH OCR STRIPPING

def clean_ocr_text(text):
    if not text:
        return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([,:;?])', r'\1\2 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ingest_pdf(pdf_path, document_id):
    loader = PyMuPDFLoader(pdf_path)  
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    
    documents = []
    for idx, chunk in enumerate(chunks):
        cleaned = clean_ocr_text(chunk.page_content)
        doc = Document(
            page_content=cleaned,
            metadata={
                "document_id": document_id,
                "source_type": "pdf",
                "source_name": os.path.basename(pdf_path), 
                "page": chunk.metadata.get("page", -1),
                "timestamp": "",
                "chunk_id": idx
            }
        )
        documents.append(doc)
    return documents

def download_audio(youtube_url, output_folder="downloads"):
    os.makedirs(output_folder, exist_ok=True)

    cookie_path = "downloads/youtube_cookies.txt"
    if "YOUTUBE_COOKIES" in st.secrets:
        with open(cookie_path, "w", encoding="utf-8") as f:
            f.write(st.secrets["YOUTUBE_COOKIES"])
            
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_folder}/%(id)s_%(title)s.%(ext)s',
        'quiet': True,
        'http_headers': {'User-Agent': 'Mozilla/5.0'}
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        audio_path = ydl.prepare_filename(info)
    return {"title": info.get("title"), "channel": info.get("uploader"), "video_id": info.get("id"), "audio_path": audio_path}

def create_timestamped_chunks(segments, chunk_size=500):
    chunks, current_text, start_time, end_time = [], "", None, None
    for segment in segments:
        text = segment.text.strip()
        if start_time is None: start_time = segment.start
        end_time = segment.end
        current_text += " " + text
        if len(current_text) >= chunk_size:
            chunks.append({"text": current_text.strip(), "start_time": round(start_time, 2), "end_time": round(end_time, 2)})
            current_text, start_time, end_time = "", None, None
    if current_text:
        chunks.append({"text": current_text.strip(), "start_time": round(start_time, 2), "end_time": round(end_time, 2)})
    return chunks

def ingest_youtube(timestamped_chunks, video_metadata, document_id):
    documents = []
    for idx, chunk in enumerate(timestamped_chunks):
        doc = Document(
            page_content=chunk["text"],
            metadata={
                "document_id": document_id,
                "source_type": "youtube",
                "source_name": video_metadata["title"],
                "video_id": video_metadata["video_id"],
                "channel_name": video_metadata["channel"],
                "page": -1,
                "timestamp": f'{chunk["start_time"]}s - {chunk["end_time"]}s',
                "chunk_id": idx
            }
        )
        documents.append(doc)
    return documents


# HELPER FOR VISION TRANSFORMATION
def encode_bytes_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


# STEP 6 — VECTOR ENGINE SETUP

def build_vectorstore():
    try:
        pc = Pinecone(api_key=pinecone_api_key)
    except Exception as e:
        st.error(f"Pinecone initialization failed: {e}")
        st.stop()

    index_name = "ai-learning-assistant"
    existing_indexes_response = pc.list_indexes()
    existing_indexes = (
        existing_indexes_response.names()
        if hasattr(existing_indexes_response, "names")
        else [index.name for index in existing_indexes_response]
    )

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return PineconeVectorStore(index_name=index_name, embedding=embedding_model, pinecone_api_key=pinecone_api_key)


# STEPS FOR CORE BACKGROUND RAG CHAINS

class RAGResponse(BaseModel):
    answer: str = Field(description="Answer generated from context")

parser = PydanticOutputParser(pydantic_object=RAGResponse)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_api_key)
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, groq_api_key=groq_api_key)

def format_docs(docs):
    formatted = ""
    for doc in docs:
        metadata = doc.metadata
        formatted += f"\n\nSOURCE ({metadata['source_type'].upper()}): {metadata['source_name']}\nCONTENT:\n{doc.page_content}"
    return formatted

pdf_prompt = ChatPromptTemplate.from_template("""
You are an expert educational AI learning assistant helping users learn from complex PDFs.

Analyze the provided context carefully and answer the user's question with deep conceptual clarity, structural elegance, and professional formatting.

CRITICAL PARSING RULE:
- You MUST respond ONLY with a valid JSON object matching the requested schema. 
- Do NOT include any introduction conversational text, headers (like ###), bullet points, or markdown text outside of the JSON block itself.
- Your entire response text must wrap directly into the raw JSON properties required below.                                 

IMPORTANT RESPONSE RULES:
- Use clean Markdown layout elements: Utilize clear headers (###), bold key phrases, and break explanations down into bullet points or numbered lists. No walls of prose.
- Connect ideas and synthesize data gracefully even if the user asks an indirect question.
- Rely ONLY on the provided context. Do not hallucinate external legal or historical facts.
- If the context completely lacks any relevant information, state exactly: "I could not find enough matching information in the provided PDF to answer that safely."

CONTEXT:
{context}

CHAT HISTORY:
{history}

QUESTION:
{question}

{format_instructions}
""")

youtube_prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant helping users master knowledge from audio lectures and video clips.

Analyze the provided transcript carefully and answer the user's question with high structural organization and professional formatting.

CRITICAL PARSING RULE:
- You MUST respond ONLY with a valid JSON object matching the requested schema. 
- Do NOT include any introduction conversational text, headers (like ###), bullet points, or markdown text outside of the JSON block itself.
- Your entire response text must wrap directly into the raw JSON properties required below.                                                 

IMPORTANT RESPONSE RULES:
- Use clean Markdown layout elements: Utilize clear headers (###), bold key phrases, and break explanations down into bullet points or numbered lists. No walls of prose.
- Synthesize ideas fluidly across timestamps to create an insightful, clear, and comprehensive answer.
- Rely ONLY on the provided context. Do not hallucinate outside details.
- If the context completely lacks any relevant information, state exactly: "I could not find enough matching information in the lecture to answer that safely."
                                                  
CONTEXT:
{context}

CHAT HISTORY:
{history}

QUESTION:
{question}

{format_instructions}
""")

contextualize_q_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and the latest user question, rewrite it into a standalone question. Do NOT answer it.
CHAT HISTORY: {history}\nQUESTION: {question}\nStandalone Question:
""")

pdf_chain = ({"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"], "format_instructions": lambda _: parser.get_format_instructions(), "history": lambda x: x.get("history", "")} | pdf_prompt | llm | parser)
youtube_chain = ({"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"], "format_instructions": lambda _: parser.get_format_instructions(), "history": lambda x: x.get("history", "")} | youtube_prompt | llm | parser)

rag_branch = RunnableBranch(
    (lambda x: len(x["docs"]) > 0 and x["docs"][0].metadata.get("source_type") == "pdf", pdf_chain),
    (lambda x: len(x["docs"]) > 0 and x["docs"][0].metadata.get("source_type") == "youtube", youtube_chain),
    pdf_chain
)

question_rewriter = contextualize_q_prompt | llm

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag = RunnableWithMessageHistory(rag_branch, get_session_history, input_messages_key="question", history_messages_key="history")

def get_contextualized_question(question, session_id="user_1"):
    history = get_session_history(session_id)
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history.messages])
    return question_rewriter.invoke({"history": history_text, "question": question}).content

# Premium Manual Hybrid Retrieval + Reranking Execution Engine
def execute_hybrid_retrieval(question, vectorstore, document_id):
    if not st.session_state.active_docs:
        return []
        
    bm25_retriever = BM25Retriever.from_documents(st.session_state.active_docs)
    bm25_retriever.k = 6
    
    if isinstance(document_id, list):
        search_filter = {
            "$or": [
                {"source_name": {"$in": document_id}},
                {"video_id": {"$in": document_id}}
            ]
        }
    else:
        search_filter = {"document_id": document_id}
    
    pinecone_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6, 
            "fetch_k": 15, 
            "lambda_mult": 0.5,
            "filter": search_filter
        }
    )
    
    bm25_docs = bm25_retriever.invoke(question)
    pinecone_docs = pinecone_retriever.invoke(question)
    
    seen, combined = set(), []
    for doc in (bm25_docs + pinecone_docs):
        if doc.page_content.strip() not in seen:
            seen.add(doc.page_content.strip())
            combined.append(doc)
            
    if not combined: return []
    
    pairs = [(question, doc.page_content) for doc in combined]
    scores = rerank_model.predict(pairs)
    reranked = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in reranked[:4]]


# HERO HEADER


st.markdown("""
<div style="text-align:center; padding-top:1rem; padding-bottom:2rem;">

<div class="hero-title">
🧠 AI Learning Assistant
</div>

<div class="hero-subtitle">
Advanced Multi-Source Retrieval-Augmented Generation System for PDFs, YouTube Lectures, and Images
</div>

</div>
""", unsafe_allow_html=True)


# SIDEBAR

with st.sidebar:

    st.markdown("## ⚙️ Workspace")

    tab1, tab2, tab3 = st.tabs(["📄 PDFs", "🎥 YouTube", "📸 Images"])

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
                st.session_state.active_image = None  # Reset complementary modes

                vstore = build_vectorstore()
                vstore.add_documents(
                    all_docs,
                    ids=[f"{i}_{datetime.now().timestamp()}" for i in range(len(all_docs))]
                )
                progress.progress(100, text="Completed!")
                st.success(f"Successfully indexed {len(all_docs)} chunks across {len(uploaded_files)} PDFs!")

    # YOUTUBE TAB
    with tab2:
        video_urls_input = st.text_area(
            "Paste YouTube URLs (comma separated)",
            key=f"yt_input_{st.session_state.yt_input_key}"
        )

        if video_urls_input and st.button("🎙️ Process Lectures"):
            progress = st.progress(0)
            with st.spinner("Transcribing YouTube lectures..."):
                urls = [url.strip() for url in video_urls_input.split(",") if url.strip()]
                all_yt_docs = []
                video_ids_list = []
                total = len(urls)

                for idx, video_url in enumerate(urls):
                    try:
                        progress.progress(int((idx / total) * 100), text=f"Processing Lecture {idx+1}/{total}")
                        meta = download_audio(video_url)
                        video_ids_list.append(meta["video_id"])
                        segments, _ = whisper_model.transcribe(meta["audio_path"])
                        ts_chunks = create_timestamped_chunks(list(segments), chunk_size=500)
                        file_docs = ingest_youtube(ts_chunks, meta, meta["video_id"])
                        all_yt_docs.extend(file_docs)
                    except Exception as e:
                        st.error(f"Failed processing: {video_url}")
                        st.exception(e)

                if all_yt_docs:
                    st.session_state.active_docs = all_yt_docs
                    st.session_state.document_id = video_ids_list
                    st.session_state.active_image = None  # Reset complementary modes

                    vstore = build_vectorstore()
                    vstore.add_documents(
                        all_yt_docs,
                        ids=[f"{i}_{datetime.now().timestamp()}" for i in range(len(all_yt_docs))]
                    )
                    progress.progress(100, text="Completed!")
                    st.success(f"Indexed {len(all_yt_docs)} timeline chunks across {len(urls)} lectures!")

    # 📸 IMAGE TAB
    with tab3:
        uploaded_image = st.file_uploader(
            "Upload Image Asset",
            type=["png", "jpg", "jpeg"],
            key=f"image_uploader_{st.session_state.image_uploader_key}"
        )

        if uploaded_image and st.button("👁️ Process Image"):
            with st.spinner("Analyzing image asset structures..."):
                # Load image byte arrays into session state parameters natively
                st.session_state.active_image = uploaded_image.getvalue()
                st.session_state.document_id = [uploaded_image.name]
                st.session_state.active_docs = [] # Wipes vector memory tracking metrics
                
                # Render preview safely inside sidebar workspace container using modern width flag
                st.image(st.session_state.active_image, caption="Active Image Preview", width=280)
                st.success(f"Successfully mounted '{uploaded_image.name}' as visual knowledge base resource!")

    
    # ACTIVE SOURCES


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
        if "user_1" in st.session_state.store:
            del st.session_state.store["user_1"]

        st.session_state.pdf_uploader_key += 1
        st.session_state.yt_input_key += 1
        st.session_state.image_uploader_key += 1
        st.success("Workspace reset complete!")
        st.rerun()


# CHAT HISTORY (FIXED & CLEAN)

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


user_query = st.chat_input(
    "Ask anything about your uploaded learning resources..."
)

if user_query:

    if not st.session_state.document_id:
        st.error("⚠️ Please upload and process resources before asking questions.")
    else:
        # USER MESSAGE
        with st.chat_message("user", avatar="👨‍🎓"):
            st.markdown(user_query)

        st.session_state.messages.append({
            "role": "user",
            "content": user_query
        })

        # ASSISTANT RESPONSE
        with st.chat_message("assistant", avatar="🤖"):

            # -----------------------------------------------------------------
            # BRANCH A: MULTIMODAL VISION PIPELINE (IMAGE LOADED)
            # -----------------------------------------------------------------
            if st.session_state.get("active_image") is not None:
                with st.spinner("🧠 Analyzing image structure and synthesizing response..."):
                    
                    # Convert the raw byte string to base64 for message payload transmission
                    base64_repr = encode_bytes_to_base64(st.session_state.active_image)
                    
                    # Force strict response constraints down prompt alongside layout guidelines
                    system_guideline = (
                        "You are an expert AI assistant specialized in analyzing visual diagrams, "
                        "charts, and real-world photos. Answer the user's question with deep clarity, "
                        "using headers (###), bold key phrases, and bullet points where structural organization helps. "
                        "Rely strictly on visual patterns observed in the provided image layout details."
                    )
                    
                    content_payload = [
                        {"type": "text", "text": f"{system_guideline}\n\nUser Question: {user_query}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_repr}"}
                        }
                    ]
                    
                    # Intercept processing from standard context chains to run visual reasoning
                    vision_msg = HumanMessage(content=content_payload)
                    raw_vision_response = vision_llm.invoke([vision_msg])
                    
                    answer_text = raw_vision_response.content
                    refs = [f"📸 Visual Context: {st.session_state.document_id[0]}"]

                # Render response card immediately
                with st.container(border=True):
                    st.markdown(answer_text)
                    
                if refs:
                    st.markdown("---")
                    st.caption("📂 Verified Sources")
                    for ref in refs:
                        st.markdown(ref)

            # -----------------------------------------------------------------
            # BRANCH B: HYBRID VECTOR RAG PIPELINE (PDF/YOUTUBE TEXT CORES)
            # -----------------------------------------------------------------
            else:
                with st.status("🔍 Running Hybrid Retrieval Pipeline...", expanded=False) as status:
                    st.write("🔄 Contextualizing conversational query...")
                    standalone = get_contextualized_question(user_query)

                    st.write("⚡ Executing BM25 + Dense Retrieval...")
                    vstore = build_vectorstore()
                    retrieved_docs = execute_hybrid_retrieval(standalone, vstore, st.session_state.document_id)

                    st.write("📊 Applying Cross-Encoder Reranking...")
                    format_insts = parser.get_format_instructions()

                    chain_input = {
                        "question": user_query,
                        "docs": retrieved_docs,
                        "history": "",
                        "format_instructions": format_insts
                    }

                    status.update(label="✅ Retrieval Complete", state="complete", expanded=False)

                with st.spinner("🧠 Synthesizing intelligent response..."):
                    response = conversational_rag.invoke(
                        chain_input,
                        config={"configurable": {"session_id": "user_1"}}
                    )

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

                        if clean_ref not in refs:
                            refs.append(clean_ref)

                    with st.container(border=True):
                        st.markdown(answer_text)

                    if refs:
                        st.markdown("---")
                        st.caption("📂 Verified Sources")
                        for ref in refs:
                            st.markdown(ref)

        # SAVE HISTORICAL MESSAGE STATE
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "references": refs
        })


