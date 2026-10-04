# DocMind AI — Intelligent Document Investigator

Ask questions about your PDFs, Word files, text files and photos of documents, and get answers that **cite the exact file, page and section**, **say how confident they are**, and **flag documents that contradict each other**.

Runs 100% locally: FastAPI + ChromaDB + Ollama (llama3.2). No API keys, no cost, your files never leave your machine.

> Built for **ALGOTHON'26 · ALG-AI-02 — Intelligent Document Investigator**

---

## What it does

| Requirement (ALG-AI-02) | How DocMind handles it |
|---|---|
| **Multiple document formats** | PDF, DOCX, TXT, MD, CSV and images (PNG, JPG, WEBP, BMP, TIFF). Images and scanned PDF pages are read with OCR. |
| **Extraction / indexing** | Text is split into chunks tagged with file, page and section heading, embedded with `all-MiniLM-L6-v2` and stored in ChromaDB. Re-uploading a file replaces it. |
| **Natural-language Q&A** | Chat interface with follow-ups ("can you elaborate?", "what about the second one?"). Small files are read in full; large ones use hybrid search (semantic + keyword). |
| **Source / section references** | Every answer has numbered citations. Clicking one opens the evidence: file, page, section and the passage itself, with your question's words highlighted. |
| **Conflict detection** | Each answer is checked for sources that disagree, and **Hunt for conflicts** compares every document in the library against the others. |
| **Uncertainty handling** | Each answer gets High / Medium / Low / Not found confidence with a reason. If nothing relevant exists, or the model's answer isn't backed by the text, it says so instead of guessing. |

### Using it like ChatGPT

- **Attach** files with the paperclip, drag and drop, or paste. Each chat remembers its own files.
- A file attached with a message becomes that message's focus. "Give me details about this" means the file you just attached.
- Every upload is kept in the **Library** sidebar; click a file there to add it to the current chat.
- A chat with no files attached searches the whole library.

---

## How it works

```
 upload ─► extractors.py ─► chunks (+ page, section) ─► embeddings ─► ChromaDB
              │  PDF: pypdf (+ OCR for scanned pages)
              │  DOCX: python-docx (headings + tables)
              │  Images: RapidOCR
              ▼
 question ─► chat's files fit in context? ── yes ─► whole text
                     │ no
                     ▼
              hybrid search (cosine + keyword boost, per-document cap)
                     ▼
            llama3.2 (JSON: answer, sources_used, confidence, conflicts)
                     ▼
     checks: grounding (answer words must appear in the sources)
             citation repair · dedicated conflict check on overlapping passages
             confidence = weaker of retrieval strength and model's own rating
                     ▼
            answer + citations + confidence + conflicts
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11 |
| LLM | Ollama · llama3.2 (local) |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Extraction | pypdf, python-docx, RapidOCR (ONNX) |
| Frontend | Single-file HTML / CSS / JS (marked + DOMPurify) |

---

## Setup

**Prerequisites:** Python 3.11 and [Ollama](https://ollama.com).

```powershell
# 1. Get the model (one time)
ollama pull llama3.2

# 2. Clone and install
git clone https://github.com/deerthika21/DocMind-rag.git
cd DocMind-rag
py -3.11 -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# 3. Run
uvicorn api:app --reload
```

Open **http://127.0.0.1:8000**. The first start takes ~20 s while the embedding model downloads.

Optional `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2        # llama3 gives better answers but is slower
```

### Troubleshooting

| Problem | Fix |
|---|---|
| "Ollama is offline" in the sidebar | Run `ollama serve` |
| "Server offline" | Start `uvicorn api:app --reload` |
| `WinError 10048` / address in use | It's already running; just open the link |
| `ModuleNotFoundError` | Activate the venv first |

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/upload` | Index files (`files`), optionally attach to a chat (`session_id`, or `new`) |
| `POST` | `/chat` | Ask a question `{question, session_id}` → answer, sources, confidence, conflicts |
| `POST` | `/conflicts/scan` | Compare all documents for contradictions |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{name}` · `/documents` | Remove one / all documents |
| `GET` | `/sessions` · `/sessions/{id}` | List chats / get one with messages and attached files |
| `POST` | `/sessions/attach` | Attach a library file to a chat |
| `DELETE` | `/sessions/{id}/documents/{name}` | Detach a file from a chat |
| `GET` | `/health` | Index size and whether Ollama is reachable |

Interactive docs: **http://127.0.0.1:8000/docs**

---

## Configuration

Tuning lives in [`config.py`](config.py):

| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 80 | Chunk length in characters |
| `FULL_CONTEXT_CHARS` | 14000 | Files up to this size are read in full instead of searched |
| `CONTEXT_K` / `MAX_PER_DOC` | 6 / 3 | Passages sent to the model, and the cap per document |
| `KEYWORD_BOOST` | 0.25 | Weight of keyword matching in hybrid search |
| `MIN_RELEVANCE` / `MEDIUM_RELEVANCE` / `HIGH_RELEVANCE` | 0.25 / 0.35 / 0.50 | Confidence thresholds (cosine similarity) |
| `CONFLICT_SIMILARITY` / `CONFLICT_MAX_PAIRS` | 0.60 / 6 | Which passages the conflict scan compares, and how many |

---

## Project structure

```
api.py           FastAPI app: indexing, retrieval, answering, confidence, conflicts, chats
extractors.py    File → text with page/section tags; OCR; chunking
config.py        Settings
frontend.html    The whole web UI
requirements.txt
src/             Earlier prototype (not used by api.py)
```

---

## Limitations

- **Small local model.** llama3.2 (3B) can be imprecise on long or messy documents; switch to `llama3` in `.env` for better quality.
- **OCR quality** depends on the photo/scan; misread names carry into answers.
- **Chats are in memory** and reset when the server restarts. Uploaded files stay in the library.
- **No login.** Don't expose it publicly with personal documents in the library.
