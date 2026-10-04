import os
import re
import json
import logging
from uuid import uuid4
from datetime import datetime
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DOCS_DIR,
    EMBEDDING_MODEL,
    FETCH_K,
    CONTEXT_K,
    MAX_PER_DOC,
    KEYWORD_BOOST,
    FULL_CONTEXT_CHARS,
    MIN_RELEVANCE,
    MEDIUM_RELEVANCE,
    HIGH_RELEVANCE,
    CONFLICT_SIMILARITY,
    CONFLICT_MAX_PAIRS,
)
from extractors import extract, chunk_segments, SUPPORTED_EXTENSIONS

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="DocMind AI - Intelligent Document Investigator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GLOBAL STATE
# =========================================================

sessions = {}

embeddings_model = None
current_vectordb = None

CONFIDENCE_LEVELS = ["none", "low", "medium", "high"]

# =========================================================
# MODELS
# =========================================================

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class SourceRef(BaseModel):
    id: int
    document: str
    page: int
    section: str
    snippet: str
    score: float
    cited: bool


class Conflict(BaseModel):
    topic: str
    details: str
    sources: list[int]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    confidence: str
    confidence_reason: str
    conflicts: list[Conflict]
    session_id: str


class DocumentOut(BaseModel):
    name: str
    type: str
    chunks: int
    pages: int
    ocr_pages: int
    uploaded_at: str


class UploadResponse(BaseModel):
    message: str
    documents: list[DocumentOut]
    errors: list[str]
    chunks_indexed: int
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    chain_ready: bool
    docs_indexed: int
    documents: int
    llm_ready: bool


class SessionOut(BaseModel):
    id: str
    title: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    attachments: list[str] = []
    sources: list[SourceRef] = []
    confidence: Optional[str] = None
    confidence_reason: Optional[str] = None
    conflicts: list[Conflict] = []


class SessionDetail(BaseModel):
    id: str
    title: str
    messages: list[MessageOut]
    documents: list[str] = []


class ConflictFinding(BaseModel):
    topic: str
    details: str
    a: SourceRef
    b: SourceRef


class ConflictScanResponse(BaseModel):
    pairs_checked: int
    conflicts: list[ConflictFinding]
    message: str

# =========================================================
# VECTOR DB HELPERS
# =========================================================

NEW_CHAT_TITLE = "New chat"


def get_or_create_session(session_id: Optional[str]):

    if not session_id or session_id not in sessions:

        session_id = session_id or str(uuid4())

        sessions[session_id] = {
            "id": session_id,
            "title": NEW_CHAT_TITLE,
            "messages": [],
            "documents": [],         # files attached to this conversation
            "pending_attachments": [],  # attached since the last question
            "updated_at": datetime.utcnow().isoformat(),
        }

    return sessions[session_id]


def count_indexed_docs():

    try:

        if current_vectordb is None:
            return 0

        return current_vectordb._collection.count()

    except Exception:
        return 0


def ref_from_meta(meta: dict):
    """Normalise chunk metadata (including chunks indexed by older versions)."""

    meta = meta or {}
    legacy = "doc_type" not in meta
    page = int(meta.get("page", 0) or 0)

    return {
        "document": os.path.basename(str(meta.get("source", "unknown"))),
        # Older index used PyPDFLoader's 0-based pages
        "page": page + 1 if legacy and "page" in meta else page,
        "section": str(meta.get("section", "") or ""),
    }


def all_chunk_metadata():

    if current_vectordb is None:
        return [], []

    data = current_vectordb._collection.get(include=["metadatas"])

    return data["ids"], data["metadatas"]


def list_documents():

    docs = {}
    _, metas = all_chunk_metadata()

    for meta in metas:

        ref = ref_from_meta(meta)
        name = ref["document"]
        meta = meta or {}

        d = docs.setdefault(name, {
            "name": name,
            "type": meta.get("doc_type", os.path.splitext(name)[1].lstrip(".")),
            "chunks": 0,
            "pages": 0,
            "ocr_pages": int(meta.get("ocr_pages", 0) or 0),
            "uploaded_at": meta.get("uploaded_at", ""),
        })

        d["chunks"] += 1
        d["pages"] = max(d["pages"], int(meta.get("total_pages", 0) or 0), ref["page"])

    return sorted(docs.values(), key=lambda d: d["uploaded_at"], reverse=True)


def delete_document_chunks(name: str):

    ids, metas = all_chunk_metadata()

    doomed = [
        cid for cid, meta in zip(ids, metas)
        if ref_from_meta(meta)["document"] == name
    ]

    if doomed:
        current_vectordb._collection.delete(ids=doomed)
        invalidate_index()

    return len(doomed)

# =========================================================
# OLLAMA
# =========================================================

def ollama_json(prompt: str, num_predict: int = 700):
    """Ask Ollama for a JSON object. Returns (parsed_or_None, raw_text)."""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
                "num_predict": num_predict,
            },
        },
        timeout=240,
    )

    response.raise_for_status()

    raw = response.json().get("response", "")

    try:
        parsed = json.loads(raw)
        return (parsed if isinstance(parsed, dict) else None), raw
    except json.JSONDecodeError:
        return None, raw


def llm_ready():

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

# =========================================================
# RETRIEVAL
# =========================================================

STOPWORDS = set("""
a an the and or but of to in on at for from by with as is are was were be been being this that these those it its
which who whom what when where why how not no yes do does did has have had can could should would may might must
will shall than then there their they them he she his her you your we our i me my about into over under also only
document documents excerpt excerpts according stated states says mentioned provided information answer
""".split())


def content_tokens(text: str):
    """Lower-cased content words and numbers ("4,000", "8.2" kept whole)."""

    text = re.sub(r"\[\d+\]", " ", text.lower())

    return {
        # crude plural folding: "certificates" ~ "certificate"
        t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") and t.isalpha() else t
        for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text)
        if t not in STOPWORDS and (len(t) >= 3 or t.isdigit())
    }


_index = None


def get_index():
    """
    In-memory copy of every chunk (text, metadata, embedding, keywords) for
    hybrid search and the conflict scan. Rebuilt after uploads/deletes.
    """

    global _index

    if _index is None:

        data = current_vectordb._collection.get(include=["embeddings", "documents", "metadatas"])

        emb = np.array(data["embeddings"], dtype=np.float32).reshape(len(data["ids"]), -1)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

        tokens = [content_tokens(t) for t in data["documents"]]
        df = {}
        for toks in tokens:
            for t in toks:
                df[t] = df.get(t, 0) + 1

        _index = {
            # chunk position within its document (ids are "name::i"; older chunks have none)
            "order": [int(cid.rsplit("::", 1)[1]) if "::" in cid and cid.rsplit("::", 1)[1].isdigit() else 0 for cid in data["ids"]],
            "texts": data["documents"],
            "refs": [ref_from_meta(m) for m in data["metadatas"]],
            "emb": emb,
            "tokens": tokens,
            "df": df,
        }

    return _index


def invalidate_index():

    global _index
    _index = None


def keyword_scores(query: str, index):
    """IDF-weighted share of the query's words found in each chunk (0..1)."""

    q_tokens = content_tokens(query)
    n = len(index["texts"])

    if not q_tokens or not n:
        return np.zeros(n, dtype=np.float32)

    # Words that appear nowhere get the maximum weight, so asking about
    # something absent from the documents keeps the score low
    idf = {t: np.log(1 + n / (index["df"].get(t, 0) + 0.5)) for t in q_tokens}
    total = sum(idf.values())

    return np.array([
        sum(w for t, w in idf.items() if t in toks) / total
        for toks in index["tokens"]
    ], dtype=np.float32)


def chunks_in_scope(scope):
    """Indexes of chunks belonging to the given documents (None = all)."""

    index = get_index()

    return [i for i, ref in enumerate(index["refs"]) if scope is None or ref["document"] in scope]


def retrieve(queries: list[str], overview: bool = False, scope=None, full: bool = False):
    """
    scope: document names to search (a conversation's attached files); None = all.

    full=True returns EVERY chunk in scope, in reading order, so the model
    sees the whole document (used when it fits in the context window).

    overview=True (questions about the documents as a whole) also includes
    the opening chunk of every document, since no single passage matches
    "what are these documents about?".

    Hybrid search: cosine similarity plus a keyword boost (catches acronyms,
    codes and names that embeddings miss). Several queries can be given
    (e.g. a follow-up with and without the previous question); each chunk
    keeps its best score. Caps chunks per document so other documents
    (and possible conflicts) make it into the context.
    """

    index = get_index()

    if not index["texts"]:
        return []

    scores = np.zeros(len(index["texts"]), dtype=np.float32)
    keywords = np.zeros(len(index["texts"]), dtype=np.float32)

    for query in queries:
        q = np.array(embeddings_model.embed_query(query), dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        dense = index["emb"] @ q
        kw = keyword_scores(query, index)
        keywords = np.maximum(keywords, kw)
        scores = np.maximum(scores, np.minimum(1.0, dense + KEYWORD_BOOST * kw))

    allowed = chunks_in_scope(scope)

    def excerpt(i):
        return {
            **index["refs"][i],
            "text": index["texts"][i],
            "emb": index["emb"][i],
            "score": round(float(scores[i]), 3),
            "keyword": float(keywords[i]),
        }

    if full:
        reading_order = sorted(allowed, key=lambda i: (index["refs"][i]["document"], index["refs"][i]["page"], index["order"][i]))
        return [excerpt(i) for i in reading_order]

    allowed_set = set(allowed)
    scores[[i for i in range(len(scores)) if i not in allowed_set]] = -1
    keywords[[i for i in range(len(keywords)) if i not in allowed_set]] = 0

    order = [int(i) for i in np.argsort(-scores)[:FETCH_K * 2] if scores[i] >= 0]
    limit = CONTEXT_K

    if overview:
        first = {}
        for i in allowed:
            ref = index["refs"][i]
            key = (ref["page"], index["order"][i])
            if ref["document"] not in first or key < first[ref["document"]][0]:
                first[ref["document"]] = (key, i)
        opening = [i for _, i in first.values()][:OVERVIEW_DOCS]
        order = opening + [i for i in order if i not in opening]
        limit = max(CONTEXT_K, len(opening))

    excerpts = []
    per_doc = {}

    # Passages containing most of the question's key terms are always
    # included, even when noisy text (OCR) embeds poorly
    keyword_hits = [int(i) for i in np.argsort(-keywords)[:2] if keywords[i] >= 0.5]
    order = keyword_hits + [i for i in order if i not in keyword_hits]
    limit += len(keyword_hits)

    for i in order:

        ref = index["refs"][i]

        if per_doc.get(ref["document"], 0) >= MAX_PER_DOC:
            continue

        per_doc[ref["document"]] = per_doc.get(ref["document"], 0) + 1

        excerpts.append(excerpt(i))

        if len(excerpts) >= limit:
            break

    return excerpts


def retrieval_confidence(top_score: float):

    if top_score >= HIGH_RELEVANCE:
        return "high"

    if top_score >= MEDIUM_RELEVANCE:
        return "medium"

    if top_score >= MIN_RELEVANCE:
        return "low"

    return "none"


def format_location(ex):

    parts = [ex["document"]]

    if ex["page"]:
        parts.append(f"page {ex['page']}")

    if ex["section"]:
        parts.append(f"section: {ex['section']}")

    return " | ".join(parts)

# =========================================================
# ANSWERING
# =========================================================

ANSWER_PROMPT = """You are a careful document investigator. Answer the QUESTION using ONLY the numbered EXCERPTS below.

Rules:
- Cite every fact with the excerpt number in square brackets, e.g. [1] or [2][3].
- Never use outside knowledge. If the excerpts do not contain the answer, say clearly that the documents do not contain it.
- Compare excerpts from different documents. If they give DIFFERENT values or contradictory claims about the same thing (numbers, dates, names, scores, requirements), report each one in "conflicts" and mention the disagreement in the answer. Different documents covering different topics is NOT a conflict.
- Set "confidence" to "high" only if the excerpts state the answer directly; "medium" if it needs inference or is partial; "low" if it is barely supported or missing.

{history}EXCERPTS:
{excerpts}

QUESTION: {question}

Respond with JSON only, in this exact shape:
{{"answer": "markdown answer with [n] citations", "sources_used": [numbers of the excerpts the answer relies on], "confidence": "high|medium|low", "confidence_reason": "one short sentence explaining the confidence", "conflicts": [{{"topic": "what they disagree about", "details": "excerpt [a] says X, excerpt [b] says Y", "sources": [a, b]}}]}}"""

NOT_FOUND_ANSWER = (
    "I couldn't find this in the uploaded documents, so I won't guess. "
    "Try rephrasing, or upload a document that covers it."
)

# Share of the answer's content words that must appear in the excerpts
MIN_GROUNDING = 0.4

# Questions about the documents as a whole ("what are these documents about?")
OVERVIEW_QUESTION = re.compile(
    r"\b(these|the|this|my|all|uploaded|each|every)\s+(documents?|files?|pdfs?|images?|uploads?)\b|\bsummar(y|ise|ize)\b|\boverview\b|\bkey points\b|\bmain (points|topics|ideas)\b",
    re.IGNORECASE,
)
OVERVIEW_DOCS = 8

# Questions that lean on the previous turn ("what about its cost?")
FOLLOW_UP_WORDS = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|he|she|his|her|there|same|above|previous|more|else|also|other|otherwise|elaborate|explain)\b",
    re.IGNORECASE,
)

NOT_FOUND_PATTERNS = re.compile(
    r"(do(es)? not (contain|mention|provide|include|specify)|no (information|mention)|not (found|mentioned|provided|available|specified)|cannot (find|determine|answer)|unable to (find|determine))",
    re.IGNORECASE,
)


CONFLICT_PROMPT = """Two passages from DIFFERENT documents discuss a similar subject.
Decide whether they CONTRADICT each other: they state different values, dates, names, scores, rules or claims about the SAME fact.
Passages that simply cover different details, or say the same thing in different words, do NOT conflict.

PASSAGE {label_a} ({loc_a}):
{text_a}

PASSAGE {label_b} ({loc_b}):
{text_b}

Respond with JSON only:
{{"conflict": true or false, "topic": "the fact they disagree on", "details": "{label_a} says ..., {label_b} says ..."}}"""


def same_text(a: str, b: str):
    return " ".join(a.split()) == " ".join(b.split())


def check_conflict(ex_a, ex_b, label_a="A", label_b="B"):
    """Ask the LLM whether two passages contradict each other. Returns {topic, details} or None."""

    parsed, _ = ollama_json(CONFLICT_PROMPT.format(
        label_a=label_a, loc_a=format_location(ex_a), text_a=ex_a["text"],
        label_b=label_b, loc_b=format_location(ex_b), text_b=ex_b["text"],
    ), num_predict=250)

    if parsed and str(parsed.get("conflict")).lower() == "true":
        return {
            "topic": str(parsed.get("topic") or "Conflicting information"),
            "details": str(parsed.get("details") or ""),
        }

    return None


def build_history(session):

    # While focused on a newly attached file, earlier turns are about other files
    start = session.get("focus_start", 0) if session.get("focus") else 0
    turns = session["messages"][start:][-4:]

    if not turns:
        return ""

    lines = []
    for m in turns:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:600]}")

    return "CONVERSATION SO FAR (for resolving follow-up questions only; not a source):\n" + "\n".join(lines) + "\n\n"


def answer_question(question: str, session):

    # Follow-ups like "what about the second one?" need the previous question
    # to retrieve well; search with and without it and keep the best matches
    queries = [question]
    prev_questions = [m["content"] for m in session["messages"] if m["role"] == "user"]
    follow_up = bool(prev_questions) and (len(question.split()) <= 3 or bool(FOLLOW_UP_WORDS.search(question)))
    if follow_up:
        queries.append(f"{prev_questions[-1]} {question}")

    # "Can you elaborate?" after an overview question is still an overview
    overview = bool(OVERVIEW_QUESTION.search(question)) or (follow_up and bool(OVERVIEW_QUESTION.search(prev_questions[-1])))

    # Like a chat app: files attached to this conversation are what it's
    # about. Without attachments, the whole library is searched.
    scope = [d for d in session.get("documents", []) if chunks_in_scope([d])] or None

    # Files attached with THIS message are what it's about ("give me details
    # about this"), and stay the focus for follow-ups like "elaborate"; a
    # standalone question goes back to all of the chat's files
    fresh = [d for d in session.get("pending_attachments", []) if chunks_in_scope([d])]
    if fresh:
        scope = fresh
        session["focus"] = fresh
        session["focus_start"] = len(session["messages"])  # history starts here
    elif follow_up and session.get("focus"):
        focus = [d for d in session["focus"] if chunks_in_scope([d])]
        scope = focus or scope
    else:
        session["focus"] = []
        session["focus_start"] = 0

    # Small enough to read in full -> give the model the whole text instead
    # of search hits (far better answers on short files like IDs, certificates)
    in_scope = chunks_in_scope(scope)
    index = get_index()
    full = sum(len(index["texts"][i]) for i in in_scope) <= FULL_CONTEXT_CHARS

    if fresh:
        queries = [question]  # the previous question was about another file

    excerpts = retrieve(queries, overview=overview, scope=scope, full=full)

    # A follow-up keeps the passages the previous answer relied on
    if follow_up and not full and not fresh:
        carried = [{**ex, "carried": True} for ex in session.get("last_excerpts", [])]
        seen = {ex["text"] for ex in carried}
        excerpts = carried + [ex for ex in excerpts if ex["text"] not in seen]
    top_score = max((ex["score"] for ex in excerpts), default=0.0)
    retrieval_level = "high" if full and excerpts else retrieval_confidence(top_score)

    # An overview is answerable from the opening passages even though no
    # single passage matches the question closely
    if overview and excerpts and CONFIDENCE_LEVELS.index(retrieval_level) < CONFIDENCE_LEVELS.index("medium"):
        retrieval_level = "medium"

    def not_found(reason):
        return {
            "answer": NOT_FOUND_ANSWER,
            "sources": [],
            "confidence": "none",
            "confidence_reason": reason,
            "conflicts": [],
        }

    # Uncertainty handling: nothing relevant -> don't let the LLM guess
    if retrieval_level == "none":
        if not any(ex["keyword"] >= 0.5 or ex.get("carried") for ex in excerpts):
            return not_found(f"No passage was relevant enough (best match {top_score:.0%}).")
        retrieval_level = "low"  # only a keyword match / carried from the previous turn

    relevant = [
        ex for ex in excerpts
        if full or overview or ex.get("carried") or ex["score"] >= MIN_RELEVANCE or ex["keyword"] >= 0.5
    ]

    if fresh:
        question += f"\n(The user just attached {', '.join(fresh)}; \"this\" / \"it\" refers to that file.)"

    if overview and len({ex["document"] for ex in relevant}) > 1:
        question += ("\n(Give more detail about EACH document, citing each one.)" if follow_up
                     else "\n(Describe briefly what EACH document is, one bullet per document, citing each one.)")

    excerpt_text = "\n\n".join(
        f"[{i}] ({format_location(ex)})\n{ex['text']}"
        for i, ex in enumerate(relevant, start=1)
    )

    prompt = ANSWER_PROMPT.format(
        # Earlier turns are about other files; they'd pull the answer back there
        history="" if fresh else build_history(session),
        excerpts=excerpt_text,
        question=question,
    )

    try:
        parsed, raw = ollama_json(prompt)
    except Exception as e:
        logger.error(f"[LLM] Error: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Language model unavailable at {OLLAMA_BASE_URL} ({e}). Is Ollama running?"
        )

    if parsed is None:
        logger.warning("[LLM] Non-JSON response; using raw text")
        parsed = {"answer": raw, "confidence": "low", "confidence_reason": "Model response was not structured.", "conflicts": []}

    answer = str(parsed.get("answer") or "").strip()

    # The model sometimes echoes the other JSON fields into the answer text
    answer = re.sub(
        r"\n+\s*\**\s*(sources?(_| )used|confidence(_| )reason|confidence|conflicts?)\s*\**\s*:.*$",
        "", answer, flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # ...or echoes the excerpt headers: "[1] (file.pdf | page 1 | section: X)"
    answer = re.sub(r"^\s*\[\d+\]\s*\([^)\n]*\|[^)\n]*\)\s*$", "", answer, flags=re.MULTILINE).strip() \
        or "The model returned an empty answer."
    n = len(relevant)
    says_not_found = bool(NOT_FOUND_PATTERNS.search(answer[:300]))

    # Grounding check: an answer whose key words aren't in any excerpt came
    # from the model's own knowledge, not the documents
    answer_tokens = content_tokens(answer)
    excerpt_tokens = [content_tokens(ex["text"]) for ex in relevant]
    all_excerpt_tokens = set().union(*excerpt_tokens)
    if answer_tokens:
        grounding = len(answer_tokens & all_excerpt_tokens) / len(answer_tokens)
    else:
        # Very short answers ("O+Ve", "A1") have no content words; match the raw text
        compact = lambda s: re.sub(r"[^a-z0-9+]", "", re.sub(r"\[\d+\]", "", s.lower()))
        grounding = 1.0 if compact(answer) and compact(answer) in compact(" ".join(ex["text"] for ex in relevant)) else 0.0

    # Overviews describe the documents ("a certificate", "an ID card") rather
    # than quote them, so word overlap says little there
    if not says_not_found and not overview and grounding < MIN_GROUNDING:
        logger.info(f"[Chat] Ungrounded answer rejected (grounding {grounding:.0%}): {answer[:120]}")
        return not_found(
            f"The model's answer wasn't supported by the documents ({grounding:.0%} of it matched), "
            f"and the best matching passage was only a {top_score:.0%} match."
        )

    # Citations: inline [n] markers, then the model's sources_used list
    cited = {int(c) for c in re.findall(r"\[(\d+)\]", answer) if 1 <= int(c) <= n}
    inline_citations = bool(cited)

    for s in parsed.get("sources_used") or []:
        try:
            s = int(str(s).strip("[] "))
        except ValueError:
            continue
        if 1 <= s <= n:
            cited.add(s)

    # Fallback: excerpts sharing distinctive words/numbers with the answer
    if not cited and not says_not_found:
        for i, toks in enumerate(excerpt_tokens, start=1):
            overlap = answer_tokens & toks
            if len(overlap) >= 3 or any(any(ch.isdigit() for ch in t) for t in overlap):
                cited.add(i)

    if cited and not inline_citations and not says_not_found:
        answer += " " + "".join(f"[{i}]" for i in sorted(cited))

    # Conflicts: keep only ones that point at real excerpts from 2+ places
    conflicts = []
    for c in parsed.get("conflicts") or []:

        if not isinstance(c, dict):
            continue

        ids = []
        for s in c.get("sources") or []:
            try:
                s = int(str(s).strip("[] "))
            except ValueError:
                continue
            if 1 <= s <= n and s not in ids:
                ids.append(s)

        # Fall back to [n] markers inside the details text
        if len(ids) < 2:
            ids = sorted({int(x) for x in re.findall(r"\[(\d+)\]", str(c.get("details", ""))) if 1 <= int(x) <= n})

        locations = {(relevant[i - 1]["document"], relevant[i - 1]["page"]) for i in ids}

        if len(ids) >= 2 and len(locations) >= 2:
            conflicts.append({
                "topic": str(c.get("topic") or "Conflicting information"),
                "details": str(c.get("details") or ""),
                "sources": ids,
            })

    # Dedicated check: strong passages from different documents that cover
    # the same thing are compared directly (the answer call alone misses some)
    strong = [(i, ex) for i, ex in enumerate(relevant, start=1) if ex["score"] >= MEDIUM_RELEVANCE][:4]
    checks = 0

    for a in range(len(strong)):
        for b in range(a + 1, len(strong)):

            (ia, ea), (ib, eb) = strong[a], strong[b]

            if checks >= 2:
                break

            if ea["document"] == eb["document"] or same_text(ea["text"], eb["text"]):
                continue

            # Only conflicts that bear on this answer
            if ia not in cited and ib not in cited:
                continue

            if any(ia in c["sources"] and ib in c["sources"] for c in conflicts):
                continue

            if float(ea["emb"] @ eb["emb"]) < CONFLICT_SIMILARITY:
                continue

            checks += 1

            try:
                found = check_conflict(ea, eb, f"[{ia}]", f"[{ib}]")
            except Exception as e:
                logger.warning(f"[Chat] Conflict check failed: {e}")
                found = None

            if found:
                conflicts.append({**found, "sources": [ia, ib]})

    # Confidence = the weaker of retrieval strength and the model's own assessment
    llm_level = str(parsed.get("confidence", "low")).lower()
    if llm_level not in CONFIDENCE_LEVELS:
        llm_level = "low"

    reasons = []
    level = min(llm_level, retrieval_level, key=CONFIDENCE_LEVELS.index)

    if parsed.get("confidence_reason"):
        reasons.append(str(parsed["confidence_reason"]).strip().rstrip(".") + ".")

    if full:
        n_docs = len({ex["document"] for ex in relevant})
        reasons.append(f"Read the full text of {n_docs} document{'s' if n_docs != 1 else ''}.")
    elif overview:
        reasons.append(f"Overview based on the opening passage of {len({ex['document'] for ex in relevant})} document(s).")
    elif retrieval_level != "high":
        reasons.append(f"Best matching passage is only a {retrieval_level} match ({top_score:.0%}).")

    if says_not_found:
        level = "low"
        reasons.append("The documents don't directly answer this.")
    elif not cited:
        level = min(level, "medium", key=CONFIDENCE_LEVELS.index)
        reasons.append("The answer doesn't cite specific passages.")

    if conflicts:
        level = min(level, "medium", key=CONFIDENCE_LEVELS.index)
        reasons.append("Sources disagree; see the conflicts below.")

    sources = [
        {
            "id": i,
            "document": ex["document"],
            "page": ex["page"],
            "section": ex["section"],
            "snippet": ex["text"][:500],
            "score": ex["score"],
            "cited": i in cited or any(i in c["sources"] for c in conflicts),
        }
        for i, ex in enumerate(relevant, start=1)
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": level,
        "confidence_reason": " ".join(reasons),
        "conflicts": conflicts,
        # Passages a follow-up question should build on
        "_excerpts": [ex for i, ex in enumerate(relevant, start=1) if i in cited] or relevant,
    }

# =========================================================
# HOME
# =========================================================

def read_frontend():

    try:

        with open("frontend.html", "r", encoding="utf-8") as f:
            return f.read()

    except OSError:

        return """
        <html>
        <body>
            <h1>DocMind AI Running</h1>
            <p>frontend.html not found.</p>
        </body>
        </html>
        """


@app.get("/", response_class=HTMLResponse)
def home():
    return read_frontend()


@app.get("/frontend.html", response_class=HTMLResponse)
def frontend():
    return read_frontend()

# =========================================================
# HEALTH
# =========================================================

@app.get("/health", response_model=HealthResponse)
def health():

    docs_count = count_indexed_docs()

    return HealthResponse(
        status="ok",
        chain_ready=docs_count > 0,
        docs_indexed=docs_count,
        documents=len(list_documents()) if docs_count else 0,
        llm_ready=llm_ready(),
    )

# =========================================================
# DOCUMENTS
# =========================================================

@app.post("/upload", response_model=UploadResponse)
def upload(files: list[UploadFile] = File(...), session_id: Optional[str] = Form(None)):
    """
    Index one or more documents. Re-uploading a file name replaces it.
    With session_id the files are attached to that conversation
    (session_id="new" starts one), which then answers from them.
    """

    if current_vectordb is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")

    os.makedirs(DOCS_DIR, exist_ok=True)

    indexed = []
    errors = []

    for file in files:

        name = os.path.basename(file.filename or "")
        ext = os.path.splitext(name)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            errors.append(f"{name}: unsupported type (supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})")
            continue

        try:

            logger.info(f"[Upload] {name}")

            file_path = os.path.join(DOCS_DIR, name)

            with open(file_path, "wb") as f:
                f.write(file.file.read())

            segments, info = extract(file_path)
            chunks = chunk_segments(segments, CHUNK_SIZE, CHUNK_OVERLAP)

            if not chunks:
                errors.append(f"{name}: no readable text found")
                continue

            replaced = delete_document_chunks(name)
            if replaced:
                logger.info(f"[Upload] Replacing {replaced} old chunks of {name}")

            uploaded_at = datetime.utcnow().isoformat()

            current_vectordb.add_texts(
                texts=[c["text"] for c in chunks],
                metadatas=[
                    {
                        "source": name,
                        "page": c["page"],
                        "section": c["section"],
                        "doc_type": info["type"],
                        "total_pages": info["pages"],
                        "ocr_pages": info["ocr_pages"],
                        "uploaded_at": uploaded_at,
                    }
                    for c in chunks
                ],
                ids=[f"{name}::{i}" for i in range(len(chunks))],
            )
            invalidate_index()

            logger.info(f"[Upload] {name}: {len(chunks)} chunks, {info}")

            indexed.append(DocumentOut(
                name=name,
                type=info["type"],
                chunks=len(chunks),
                pages=info["pages"],
                ocr_pages=info["ocr_pages"],
                uploaded_at=uploaded_at,
            ))

        except Exception as e:

            logger.error(f"[Upload] {name} failed: {e}", exc_info=True)
            errors.append(f"{name}: {e}")

    if not indexed:
        raise HTTPException(status_code=400, detail="; ".join(errors) or "No files uploaded")

    attached_to = None

    if session_id is not None:
        session = get_or_create_session(None if session_id == "new" else session_id)
        for doc in indexed:
            for key in ("documents", "pending_attachments"):
                if doc.name not in session[key]:
                    session[key].append(doc.name)
        session["updated_at"] = datetime.utcnow().isoformat()
        attached_to = session["id"]

    return UploadResponse(
        message=f"Indexed {len(indexed)} document{'s' if len(indexed) != 1 else ''}",
        documents=indexed,
        errors=errors,
        chunks_indexed=count_indexed_docs(),
        session_id=attached_to,
    )


@app.get("/documents", response_model=list[DocumentOut])
def documents():
    return [DocumentOut(**d) for d in list_documents()]


@app.delete("/documents/{name}")
def delete_document(name: str):

    removed = delete_document_chunks(name)

    if not removed:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"message": f"Removed {name}", "chunks_removed": removed}


@app.delete("/documents")
def delete_all_documents():

    ids, _ = all_chunk_metadata()

    if ids:
        current_vectordb._collection.delete(ids=ids)
        invalidate_index()

    return {"message": "All documents removed", "chunks_removed": len(ids)}

# =========================================================
# CHAT
# =========================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if count_indexed_docs() == 0:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    session = get_or_create_session(request.session_id)
    session_id = session["id"]

    if session["title"] == NEW_CHAT_TITLE:
        session["title"] = question[:40]

    logger.info(f"[Chat] Question: {question} (attached: {session['documents'] or 'all documents'})")

    try:
        result = answer_question(question, session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Chat] ERROR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    session["last_excerpts"] = result.pop("_excerpts", [])
    session["messages"].append({"role": "user", "content": question, "attachments": session["pending_attachments"]})
    session["pending_attachments"] = []
    session["messages"].append({"role": "assistant", "content": result["answer"], **{k: v for k, v in result.items() if k != "answer"}})
    session["updated_at"] = datetime.utcnow().isoformat()

    return ChatResponse(**result, session_id=session_id)

# =========================================================
# CONFLICT SCAN (across all documents)
# =========================================================

@app.post("/conflicts/scan", response_model=ConflictScanResponse)
def scan_conflicts():
    """Find passages in different documents that cover the same thing, and check them for contradictions."""

    if count_indexed_docs() == 0:
        raise HTTPException(status_code=400, detail="Please upload documents first")

    index = get_index()
    refs = index["refs"]
    texts = index["texts"]
    names = np.array([r["document"] for r in refs])

    if len(set(names)) < 2:
        return ConflictScanResponse(pairs_checked=0, conflicts=[], message="Upload at least two documents to compare.")

    sim = index["emb"] @ index["emb"].T
    sim[names[:, None] == names[None, :]] = -1  # only compare across documents
    sim = np.triu(sim, k=1)

    rows, cols = np.where(sim >= CONFLICT_SIMILARITY)
    order = np.argsort(-sim[rows, cols])

    pairs = []
    seen = set()

    for idx in order:

        i, j = int(rows[idx]), int(cols[idx])

        # Identical text can't conflict
        if same_text(texts[i], texts[j]):
            continue

        # One pair per document pair + section, so one topic doesn't eat the budget
        key = tuple(sorted([(refs[i]["document"], refs[i]["section"]), (refs[j]["document"], refs[j]["section"])]))
        if key in seen:
            continue
        seen.add(key)

        pairs.append((i, j, float(sim[i, j])))

        if len(pairs) >= CONFLICT_MAX_PAIRS:
            break

    findings = []

    for i, j, score in pairs:

        def src(k, sid):
            return SourceRef(id=sid, **refs[k], snippet=texts[k][:500], score=round(score, 3), cited=True)

        try:
            found = check_conflict(
                {**refs[i], "text": texts[i]},
                {**refs[j], "text": texts[j]},
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Language model unavailable: {e}")

        if found:
            findings.append(ConflictFinding(**found, a=src(i, 1), b=src(j, 2)))

    if not pairs:
        message = "No overlapping passages between documents, so nothing to compare."
    elif findings:
        message = f"Found {len(findings)} conflict{'s' if len(findings) != 1 else ''} in {len(pairs)} overlapping passage pairs."
    else:
        message = f"Checked {len(pairs)} overlapping passage pairs; no contradictions found."

    return ConflictScanResponse(pairs_checked=len(pairs), conflicts=findings, message=message)

# =========================================================
# SESSIONS
# =========================================================

@app.get("/sessions", response_model=list[SessionOut])
def list_sessions():

    return [
        SessionOut(
            id=s["id"],
            title=s["title"],
            updated_at=s["updated_at"]
        )
        for s in sorted(sessions.values(), key=lambda s: s["updated_at"], reverse=True)
    ]


@app.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str):

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    s = sessions[session_id]

    return SessionDetail(
        id=s["id"],
        title=s["title"],
        messages=[MessageOut(**m) for m in s["messages"]],
        documents=s.get("documents", []),
    )


class AttachRequest(BaseModel):
    name: str
    session_id: Optional[str] = None


@app.post("/sessions/attach")
def attach_document(request: AttachRequest):
    """Attach an already-indexed document to a conversation (new one if no session_id)."""

    if not chunks_in_scope([request.name]):
        raise HTTPException(status_code=404, detail="Document not found")

    session = get_or_create_session(request.session_id)

    for key in ("documents", "pending_attachments"):
        if request.name not in session[key]:
            session[key].append(request.name)

    return {"session_id": session["id"], "documents": session["documents"]}


@app.delete("/sessions/{session_id}/documents/{name}")
def detach_document(session_id: str, name: str):

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    s = sessions[session_id]
    s["documents"] = [d for d in s.get("documents", []) if d != name]
    s["pending_attachments"] = [d for d in s.get("pending_attachments", []) if d != name]

    return {"session_id": session_id, "documents": s["documents"]}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):

    sessions.pop(session_id, None)

    return {"message": "Session deleted"}

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():

    global current_vectordb
    global embeddings_model

    logger.info("[API] Starting DocMind AI...")

    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    embeddings_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    current_vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME
    )

    logger.info(f"[Startup] Vector DB loaded ({count_indexed_docs()} chunks)")
    logger.info("[API] Ready!")


@app.on_event("shutdown")
def shutdown():

    sessions.clear()

    logger.info("[API] Shutdown complete")
