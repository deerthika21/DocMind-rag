import os
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_DIR,
    COLLECTION_NAME
)

logger = logging.getLogger(__name__)


def ingest_pdf(file_path, embeddings):
    """
    Ingest PDF into Chroma vector DB
    """

    try:
        logger.info(f"[Ingest] Starting: {file_path}")

        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        logger.info(f"[Ingest] Loaded {len(documents)} pages")

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        docs = splitter.split_documents(documents)

        logger.info(f"[Ingest] Created {len(docs)} chunks")

        # Open existing Chroma DB
        vectordb = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )

        # DELETE OLD DOCUMENTS OF SAME FILE
        try:
            existing = vectordb.get(
                where={"source": file_path}
            )

            if existing and existing.get("ids"):
                vectordb.delete(ids=existing["ids"])

                logger.info(
                    f"[Ingest] Removed {len(existing['ids'])} old chunks"
                )

        except Exception as e:
            logger.warning(f"[Ingest] Delete warning: {e}")

        # Add new documents
        vectordb.add_documents(docs)

        logger.info(f"[Ingest] Indexed {len(docs)} chunks")

        return True

    except Exception as e:
        logger.error(f"[Ingest] Error: {e}", exc_info=True)
        return False