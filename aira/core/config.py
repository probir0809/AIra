from pathlib import Path
import torch

# Model configuration
MODEL_NAME = "Qwen/Qwen3-4B"  # HuggingFace model
DEVICE = DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 2048
TEMPERATURE = 0.7   #0 = deterministic, 1 = very creative.
TOP_P = 0.6       # 0.1 more determinastic, 0.9 = more creative.

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # HuggingFace model sentence transformer


# paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

# Ensure model directory exists
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# RAG Chunking Configuration
SEMANTIC_CHUNK_MIN_SIZE = 500       
SEMANTIC_CHUNK_MAX_SIZE = 1500      
SEMANTIC_BREAKPOINT_THRESHOLD = 0.75  

# FAISS Index path
FAISS_INDEX_PATH = "data/faiss"
RAG_DOC = "/home/gaian/Desktop/MLOPs/rag"

# Reranker
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 5       # final docs passed to LLM after reranking

# Retrieval (increase coarse fetch so reranker has more to work with)
RETRIEVER_TOP_K = 20     # initial FAISS fetch before reranking
