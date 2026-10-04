import os
from dotenv import load_dotenv
load_dotenv()

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

# Embeddings — 90 MB, fast, no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "rag_documents"

# Disable telemetry (stops error spam)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

# Docs folder
DOCS_DIR        = "./docs"

# Chunking
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 80

# Retrieval
RETRIEVER_K     = 3
BM25_WEIGHT     = 0.4
DENSE_WEIGHT    = 0.6
FETCH_K         = 15    # candidates pulled from the vector DB
CONTEXT_K       = 6     # excerpts sent to the LLM
MAX_PER_DOC     = 3     # cap per document so other documents (and conflicts) surface
KEYWORD_BOOST   = 0.25  # hybrid search: added to cosine score x share of query words matched
FULL_CONTEXT_CHARS = 14000  # documents this small are read in full instead of searched

# Uncertainty (cosine similarity of best match, all-MiniLM-L6-v2)
MIN_RELEVANCE    = 0.25  # below this nothing relevant was found -> refuse to answer
MEDIUM_RELEVANCE = 0.35
HIGH_RELEVANCE   = 0.50

# Cross-document conflict scan
CONFLICT_SIMILARITY = 0.60  # passages this similar are about the same thing
CONFLICT_MAX_PAIRS  = 6     # LLM checks per scan

# Memory
MEMORY_WINDOW   = 5

# SQLite
SQLITE_URL      = "sqlite+aiosqlite:///./chat_history.db"