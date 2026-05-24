from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage

from backend.app.config import GROQ_API_KEY
from backend.app.models.schemas import WorkspaceSession
from backend.app.models.pydantic_llm import RAGResponse, QuizSchema
from backend.app.utils.helpers import format_docs, encode_bytes_to_base64

# Output Parsers
parser = PydanticOutputParser(pydantic_object=RAGResponse)
quiz_parser = PydanticOutputParser(pydantic_object=QuizSchema)

# Groq LLMs Setup
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)
vision_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, groq_api_key=GROQ_API_KEY)

# RAG Prompts
pdf_prompt = ChatPromptTemplate.from_template(
    "You are Curio AI, an expert educational learning assistant helping users learn from complex PDFs. "
    "Analyze the provided context carefully and answer the user's question with deep conceptual clarity, "
    "structural elegance, and professional formatting.\n\n"
    "CRITICAL PARSING RULE:\n"
    "- You MUST respond ONLY with a valid JSON object matching the requested schema.\n"
    "- Do NOT include any introduction conversational text, headers (like ###), bullet points, or markdown text outside of the JSON block itself.\n"
    "- Your entire response text must wrap directly into the raw JSON properties required below.\n\n"
    "IMPORTANT RESPONSE RULES:\n"
    "- Use clean Markdown layout elements: Utilize clear headers (###), bold key phrases, and break explanations down into bullet points or numbered lists. No walls of prose.\n"
    "- Connect ideas and synthesize data gracefully even if the user asks an indirect question.\n"
    "- Rely ONLY on the provided context. Do not hallucinate external legal or historical facts.\n"
    "- If the context completely lacks any relevant information, state exactly: 'I could not find enough matching information in the provided sources to answer that safely.'\n\n"
    "CONTEXT: {context}\n"
    "CHAT HISTORY: {history}\n"
    "QUESTION: {question}\n"
    "{format_instructions}"
)

youtube_prompt = ChatPromptTemplate.from_template(
    "You are Curio AI, an expert educational learning assistant helping users master knowledge from audio lectures and video clips. "
    "Analyze the provided transcript(s) carefully and answer the user's question with high structural organization "
    "and professional formatting.\n\n"
    "CRITICAL PARSING RULE:\n"
    "- You MUST respond ONLY with a valid JSON object matching the requested schema.\n"
    "- Do NOT include any introduction conversational text, headers (like ###), bullet points, or markdown text outside of the JSON block itself.\n"
    "- Your entire response text must wrap directly into the raw JSON properties required below.\n\n"
    "IMPORTANT RESPONSE RULES:\n"
    "- Use clean Markdown layout elements: Utilize clear headers (###), bold key phrases, and break explanations down into bullet points or numbered lists. No walls of prose.\n"
    "- Synthesize ideas fluidly across timestamps and sources to create an insightful, clear, and comprehensive answer.\n"
    "- Rely ONLY on the provided context. Do not hallucinate outside details.\n"
    "- If the context completely lacks any relevant information, state exactly: 'I could not find enough matching information in the provided sources to answer that safely.'\n\n"
    "CONTEXT: {context}\n"
    "CHAT HISTORY: {history}\n"
    "QUESTION: {question}\n"
    "{format_instructions}"
)

contextualize_q_prompt = ChatPromptTemplate.from_template(
    "Given the conversation history and the latest user question, rewrite it into a standalone question. "
    "Do NOT answer it.\n\n"
    "CHAT HISTORY: {history}\n"
    "QUESTION: {question}\n"
    "Standalone Question:"
)

# LLM Chain Configurations
pdf_chain = (
    {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"], "format_instructions": lambda _: parser.get_format_instructions(), "history": lambda x: x.get("history", "")}
    | pdf_prompt 
    | llm 
    | parser
)

youtube_chain = (
    {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"], "format_instructions": lambda _: parser.get_format_instructions(), "history": lambda x: x.get("history", "")}
    | youtube_prompt 
    | llm 
    | parser
)

rag_branch = RunnableBranch(
    (lambda x: len(x["docs"]) > 0 and x["docs"][0].metadata.get("source_type") == "pdf", pdf_chain), 
    (lambda x: len(x["docs"]) > 0 and x["docs"][0].metadata.get("source_type") == "youtube", youtube_chain), 
    pdf_chain
)

question_rewriter = contextualize_q_prompt | llm

# Session Chat Memory Functions
def get_session_history(session_tracker_id: str, ws_session: WorkspaceSession):
    """Retrieve or initialize standard InMemoryChatMessageHistory for a workspace session."""
    if session_tracker_id not in ws_session.store: 
        ws_session.store[session_tracker_id] = InMemoryChatMessageHistory()
    return ws_session.store[session_tracker_id]

async def get_contextualized_question(user_query: str, ws_session: WorkspaceSession, session_tracker_id="user_1") -> str:
    """Rewrite a user query contextually using chat history to support follow-up questions."""
    chat_log = get_session_history(session_tracker_id, ws_session)
    chat_log_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_log.messages])
    if not chat_log_text:
        return user_query
    try:
        response = await question_rewriter.ainvoke({"history": chat_log_text, "question": user_query})
        return response.content
    except Exception:
        return user_query

# RAG Executor
async def run_conversational_rag(chain_input: dict, ws_session: WorkspaceSession, session_id="user_1"):
    """Orchestrate conversational retrieval chain matching history requirements."""
    conversational_rag = RunnableWithMessageHistory(
        rag_branch, 
        lambda sid: get_session_history(sid, ws_session), 
        input_messages_key="question", 
        history_messages_key="history"
    )
    return await conversational_rag.ainvoke(chain_input, config={"configurable": {"session_id": session_id}})

def get_balanced_context(active_docs: list, max_total_chunks: int = 32) -> str:
    """Group active document chunks by source and interleave them to prevent source starvation."""
    if not active_docs:
        return "No active resources loaded."
    
    docs_by_source = {}
    for doc in active_docs:
        # Group by document_id to guarantee unique source grouping
        doc_id = doc.metadata.get("document_id", "Unknown")
        if doc_id not in docs_by_source:
            docs_by_source[doc_id] = []
        docs_by_source[doc_id].append(doc)
        
    num_sources = len(docs_by_source)
    if num_sources == 0:
        return "No active resources loaded."
        
    chunks_per_source = max(2, max_total_chunks // num_sources)
    
    # Pre-sample doc_list for each source to be evenly distributed across its length
    sampled_docs_by_source = {}
    for doc_id, doc_list in docs_by_source.items():
        N = len(doc_list)
        if N <= chunks_per_source:
            sampled_docs_by_source[doc_id] = doc_list
        else:
            indices = [int(j * (N - 1) / (chunks_per_source - 1)) for j in range(chunks_per_source)]
            # De-duplicate indices while preserving order
            seen_idx = set()
            sampled_list = []
            for idx in indices:
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    sampled_list.append(doc_list[idx])
            sampled_docs_by_source[doc_id] = sampled_list
            
    balanced_docs = []
    
    # Interleave chunks: take 1st chunk of each, then 2nd of each, etc.
    max_chunks_in_any_source = max(len(lst) for lst in sampled_docs_by_source.values())
    for i in range(max_chunks_in_any_source):
        for doc_id, doc_list in sampled_docs_by_source.items():
            if i < len(doc_list):
                balanced_docs.append(doc_list[i])
        if len(balanced_docs) >= max_total_chunks:
            balanced_docs = balanced_docs[:max_total_chunks]
            break
            
    return "\n".join([f"SOURCE: {d.metadata.get('source_name', 'Unknown')} -> {d.page_content}" for d in balanced_docs])

# --- Orchestrated Workspace Action Service Functions ---

async def generate_summary_service(ws: WorkspaceSession) -> str:
    """Generate high-density conceptual summary from text assets or visual preview."""
    is_image_mode = ws.active_image is not None
    
    if not is_image_mode:
        context_snippet = get_balanced_context(ws.active_docs, max_total_chunks=16)
        summary_prompt = (
            "You are an elite academic compiler. Synthesize a clean, professional, high-density structured "
            "core study summary of the active resource material provided below. "
            "Utilize clean markdown headers (###), bold metrics, and succinct bullet points.\n\n"
            f"MATERIAL CONTEXT:\n{context_snippet}"
        )
        response = await llm.ainvoke(summary_prompt)
        ws.active_summary_data = response.content
    else:
        base64_repr = encode_bytes_to_base64(ws.active_image)
        vision_payload = [
            {
                "type": "text", 
                "text": "You are an elite academic compiler. Closely analyze this image file and synthesize a clean, "
                        "comprehensive, high-density study summary explaining all concepts, questions, or problems "
                        "visible within it. Use clear headers (###), bold key terms, and explicit bullet points."
            },
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_repr}"}}
        ]
        response = await vision_llm.ainvoke([HumanMessage(content=vision_payload)])
        ws.active_summary_data = response.content
        
    ws.active_quiz_data = None
    ws.active_comparison_data = None
    return ws.active_summary_data

async def generate_quiz_service(ws: WorkspaceSession) -> dict:
    """Orchestrate custom educational multiple choice quiz questions based on loaded assets."""
    is_image_mode = ws.active_image is not None
    
    if not is_image_mode:
        context_snippet = get_balanced_context(ws.active_docs, max_total_chunks=16)
        quiz_prompt = (
            "You are a university professor. Generate 3 high-quality multiple choice questions testing critical "
            "concepts explicitly covered in the context below. You MUST respond strictly with a valid JSON format "
            f"adhering to the following instructions:\n{quiz_parser.get_format_instructions()}\n\n"
            f"MATERIAL CONTEXT:\n{context_snippet}"
        )
        response = await llm.ainvoke(quiz_prompt)
        parsed_quiz = quiz_parser.parse(response.content)
        ws.active_quiz_data = parsed_quiz.dict()
    else:
        base64_repr = encode_bytes_to_base64(ws.active_image)
        quiz_prompt = (
            "You are a university professor. Analyze this image and generate 3 high-quality multiple choice "
            "questions evaluating the concepts or solving the type of problem sets visible in the asset. "
            "You MUST respond strictly with a valid JSON format adhering to the following instructions:\n"
            f"{quiz_parser.get_format_instructions()}"
        )
        vision_payload = [
            {"type": "text", "text": quiz_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_repr}"}}
        ]
        response = await vision_llm.ainvoke([HumanMessage(content=vision_payload)])
        parsed_quiz = quiz_parser.parse(response.content)
        ws.active_quiz_data = parsed_quiz.dict()
        
    ws.active_summary_data = None
    ws.active_comparison_data = None
    return ws.active_quiz_data

async def generate_comparison_service(ws: WorkspaceSession) -> str:
    """Compare multiple text assets into a detailed matrix array."""
    if ws.active_image is not None:
        raise ValueError("Comparison matrices are optimized for textual datasets, not single image assets.")
    if len(ws.document_ids) < 2:
        raise ValueError("Comparative evaluations yield optimal metrics when multiple files are loaded together.")
        
    context_snippet = get_balanced_context(ws.active_docs, max_total_chunks=16)
    matrix_prompt = (
        "You are an expert technical auditor. Provide a comprehensive cross-comparison matrix of the core "
        "arguments and insights across the loaded files. You MUST present your comparison inside a clean, "
        "comprehensive Markdown evaluation grid table with detailed column fields mapping similarities, "
        "distinct gaps, and conceptual differences.\n\n"
        f"MATERIAL CONTEXTS FROM LOADED ACTIVE FILES:\n{context_snippet}"
    )
    
    response = await llm.ainvoke(matrix_prompt)
    ws.active_comparison_data = response.content
    
    ws.active_quiz_data = None
    ws.active_summary_data = None
    return ws.active_comparison_data
