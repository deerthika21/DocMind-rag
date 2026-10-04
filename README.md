# DocMind AI — RAG Chatbot

A production-grade Retrieval-Augmented Generation chatbot built with
FastAPI, LangChain, ChromaDB, and Ollama (100% local, no API cost).

## Features
- PDF upload + semantic chunking
- Hybrid BM25 + dense vector retrieval
- Local LLM via Ollama (llama3.2)
- Conversational memory (follow-up questions work)
- Persistent chat history (SQLite)
- Dark futuristic UI with session management

## Tech Stack
| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.11 |
| LLM | Ollama (llama3.2) — local, free |
| Vector DB | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Framework | LangChain |
| Frontend | HTML / CSS / JS |
| Database | SQLite (aiosqlite) |

## Setup
```bash
# 1. Install Ollama from https://ollama.ai
ollama pull llama3.2

# 2. Clone and setup
git clone https://github.com/YOUR_USERNAME/rag-chatbot
cd rag-chatbot
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 3. Start backend
uvicorn api:app --reload --port 8000

# 4. Open frontend.html in browser (VS Code Live Server)
```

## Architecture