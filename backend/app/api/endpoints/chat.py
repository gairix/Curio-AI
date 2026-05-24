import re
import asyncio
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from backend.app.models.schemas import get_session, ChatQuery, ActionRequest
from backend.app.utils.helpers import encode_bytes_to_base64
from backend.app.services.retrieval import build_vectorstore, execute_hybrid_retrieval
from backend.app.services.llm import (
    get_contextualized_question, 
    run_conversational_rag, 
    parser,
    get_session_history,
    vision_llm
)

router = APIRouter()

@router.post("/chat")
async def chat_interaction(query_data: ChatQuery):
    """Process a chat inquiry over active resources (text or multimodal image visual RAG)."""
    ws = get_session(query_data.session_id)
    if not ws.document_ids:
        raise HTTPException(
            status_code=400, 
            detail="Please upload or process learning resources before asking questions."
        )
        
    user_query = query_data.query
    ws.messages.append({"role": "user", "content": user_query, "references": []})
    
    try:
        # --- Multimodal Image RAG Mode ---
        if ws.active_image is not None:
            base64_repr = encode_bytes_to_base64(ws.active_image)
            system_guideline = (
                "You are an expert AI assistant specialized in analyzing visual diagrams, charts, and photos. "
                "Answer with deep clarity, using headers (###), bold items, and lists. "
                "Rely strictly on visual patterns observed in the image layout details."
            )
            content_payload = [
                {"type": "text", "text": f"{system_guideline}\n\nUser Question: {user_query}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_repr}"}}
            ]
            
            # Send context directly to vision LLM (wrapped in run_in_executor to keep FastAPI loop fully non-blocking)
            loop = asyncio.get_event_loop()
            raw_vision_response = await loop.run_in_executor(
                None, 
                lambda: vision_llm.invoke([HumanMessage(content=content_payload)])
            )
            answer_text = raw_vision_response.content
            refs = [f"📸 Visual Context: {ws.document_ids[0]}"]
            
        # --- Standard Context Text RAG Mode ---
        else:
            # Standalone contextual rewrite of prompt
            standalone = await get_contextualized_question(user_query, ws, "user_1")
            
            # Execute BM25 + Dense Semantic retrieval reranked via Cross-Encoder
            loop = asyncio.get_event_loop()
            vstore = build_vectorstore()
            retrieved_docs = await loop.run_in_executor(
                None,
                lambda: execute_hybrid_retrieval(standalone, vstore, ws.document_ids, ws.active_docs)
            )
            
            format_insts = parser.get_format_instructions()
            chain_input = {
                "question": user_query, 
                "docs": retrieved_docs, 
                "history": "", 
                "format_instructions": format_insts
            }
            
            # Orchestrate response via conversational LLM chain
            response = await run_conversational_rag(chain_input, ws, "user_1")
            answer_text = response.answer if hasattr(response, 'answer') else str(response)
            
            # Gather references citation strings
            refs = []
            for doc in retrieved_docs:
                m = doc.metadata
                if m['source_type'] == 'pdf':
                    clean_ref = f"📄 {m['source_name']} (Page {int(m['page']) + 1})"
                elif m['source_type'] == 'media_file':
                    clean_ref = f"🎙️ {m['source_name']} ({m['timestamp']})"
                else:
                    raw_timestamp = str(m.get("timestamp", "0"))
                    match = re.search(r"\d+(?:\.\d+)?", raw_timestamp)
                    start_seconds = int(float(match.group())) if match else 0
                    video_id = m.get('video_id', '')
                    clickable_url = f"https://www.youtube.com/watch?v={video_id}&t={start_seconds}s"
                    clean_ref = f"🎥 {m['source_name']} [({m['timestamp']})]({clickable_url})"
                
                if clean_ref not in refs: 
                    refs.append(clean_ref)
                    
        # Do not supply reference citations if the agent states it cannot find the answer
        if "I could not find enough matching information" in answer_text:
            refs = []
            
        # Log chatbot responses
        ws.messages.append({"role": "assistant", "content": answer_text, "references": refs})
        return {"answer": answer_text, "references": refs}
        
    except Exception as e:
        # Wipe user prompt if LLM failures occur
        if ws.messages and ws.messages[-1]["role"] == "user":
            ws.messages.pop()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history")
async def get_chat_history(session_id: str = "default"):
    """Retrieve user and agent conversation logs for front-end rendering."""
    ws = get_session(session_id)
    return {"messages": ws.messages}

@router.post("/clear-history")
async def clear_history(payload: ActionRequest):
    """Clear chat conversation memory logs for the specified session."""
    ws = get_session(payload.session_id)
    ws.messages = []
    if "user_1" in ws.store:
        del ws.store["user_1"]
    return {"status": "success", "message": "Chat history cleared!"}
