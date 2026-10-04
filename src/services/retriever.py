import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHROMA_DIR, COLLECTION_NAME, RETRIEVER_K, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def build_retriever():
    """Build retriever from vector store"""
    try:
        logger.info(f"[Retriever] Building from {CHROMA_DIR}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )

        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        retriever = vectordb.as_retriever(
            search_kwargs={"k": RETRIEVER_K}
        )
        
        logger.info(f"[Retriever] Ready with k={RETRIEVER_K}")
        return retriever
        
    except Exception as e:
        logger.error(f"[Retriever] Error: {e}")
        return None