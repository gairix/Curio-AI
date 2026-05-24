from typing import List
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from backend.app.config import PINECONE_API_KEY
from backend.app.core.models import text_vectorizer, relevance_scorer

def build_vectorstore() -> PineconeVectorStore:
    """Initialize Pinecone client and setup connection to the indexes."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable is missing.")
        
    pc_instance = Pinecone(api_key=PINECONE_API_KEY)
    db_namespace = "ai-learning-assistant"
    active_db_response = pc_instance.list_indexes()
    
    # Extract index names safely
    active_db_list = (
        active_db_response.names() 
        if hasattr(active_db_response, "names") 
        else [idx_obj.name for idx_obj in active_db_response]
    )
    
    # Create the index if not already present
    if db_namespace not in active_db_list: 
        pc_instance.create_index(
            name=db_namespace, 
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
    return PineconeVectorStore(
        index_name=db_namespace, 
        embedding=text_vectorizer, 
        pinecone_api_key=PINECONE_API_KEY
    )

def execute_hybrid_retrieval(
    user_query: str, 
    db_vectorstore: PineconeVectorStore, 
    doc_reference_ids: List[str], 
    active_docs: List[Document]
) -> List[Document]:
    """Execute keyword (BM25) + vector (Pinecone) MMR retrieval and rerank using Cross-Encoder."""
    if not active_docs: 
        return []
    
    # Scale retrieval parameters based on the number of active documents
    num_docs = len(doc_reference_ids) if isinstance(doc_reference_ids, list) else 1
    k_lexical = 6 if num_docs <= 1 else min(12, 6 * num_docs)
    k_semantic = 6 if num_docs <= 1 else min(12, 6 * num_docs)
    fetch_k_semantic = 15 if num_docs <= 1 else min(30, 15 * num_docs)
    k_returned = 4 if num_docs <= 1 else min(10, 4 * num_docs)
    
    # 1. Lexical Retrieval (Keyword Matching)
    lexical_retriever = BM25Retriever.from_documents(active_docs)
    lexical_retriever.k = k_lexical
    
    # 2. Semantic Retrieval (Vector Similarity)
    search_criteria = (
        {"document_id": {"$in": doc_reference_ids}} 
        if isinstance(doc_reference_ids, list) 
        else {"document_id": doc_reference_ids}
    )
    semantic_retriever = db_vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": k_semantic, "fetch_k": fetch_k_semantic, "lambda_mult": 0.5, "filter": search_criteria}
    )
    
    # Execute retrievals
    try:
        lexical_results = lexical_retriever.invoke(user_query)
    except Exception:
        lexical_results = []
        
    try:
        semantic_results = semantic_retriever.invoke(user_query)
    except Exception:
        semantic_results = []
    
    # 3. Deduplicate Combined Results
    tracked_strings, merged_results = set(), []
    for knowledge_doc in (lexical_results + semantic_results):
        doc_content = knowledge_doc.page_content.strip()
        if doc_content not in tracked_strings:
            tracked_strings.add(doc_content)
            merged_results.append(knowledge_doc)
            
    if not merged_results: 
        return []
    
    # 4. Cross-Encoder Re-ranking
    eval_pairs = [(user_query, knowledge_doc.page_content) for knowledge_doc in merged_results]
    scoring_weights = relevance_scorer.predict(eval_pairs)
    
    sorted_eval = sorted(zip(merged_results, scoring_weights), key=lambda x: x[1], reverse=True)
    return [knowledge_doc for knowledge_doc, weight in sorted_eval[:k_returned]]
