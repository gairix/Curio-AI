from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from faster_whisper import WhisperModel

print("Initializing Machine Learning Models (SentenceTransformer, CrossEncoder, Whisper)...")

# 1. Text Vectorizer (Semantic Embedding Search)
text_vectorizer = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"}, 
    encode_kwargs={"normalize_embeddings": True}
)

# 2. Relevance Scorer (Cross-Encoder Re-ranker)
relevance_scorer = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# 3. Whisper Model (Local Transcription Engine)
whisper_model = WhisperModel("tiny", device="cpu", compute_type="float32")

print("ML Models Initialized Successfully.")
