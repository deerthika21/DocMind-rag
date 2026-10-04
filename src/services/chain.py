import logging
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, MEMORY_WINDOW

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONDENSE_PROMPT = ChatPromptTemplate.from_template("""
Given the conversation history and a follow-up question,
rewrite the follow-up into a standalone question.
If already standalone, return it unchanged. Do NOT answer.

Chat history: {chat_history}
Follow-up: {question}
Standalone question:""")

ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are a helpful document assistant.

RULES:
1. Answer ONLY from the CONTEXT below.
2. If the answer is not in context, say: "I don't know based on the uploaded documents."
3. Never guess or use outside knowledge.
4. Always cite: [Source: filename, page N]
5. Use bullet points for multi-part answers.

CONTEXT:
{context}

HISTORY:
{chat_history}

QUESTION: {question}

ANSWER:""")

_chains: dict = {}


def get_or_create_chain(session_id: str, retriever):
    if session_id not in _chains:
        logger.info(f"[Chain] Building for session {session_id}")
        _chains[session_id] = _build(retriever)
    return _chains[session_id]


def drop_chain(session_id: str):
    _chains.pop(session_id, None)


def _build(retriever):
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0
    )

    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"       # CRITICAL — must match chain output_key
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",      # CRITICAL — prevents "No response" bug
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
        verbose=False
    )
    logger.info("[Chain] Ready")
    return chain


def invoke_chain(chain, question: str) -> tuple:
    if not question or not question.strip():
        raise ValueError("Question is empty")

    logger.info(f"[Chain] Q: {question[:70]}")
    result = chain.invoke({"question": question})
    logger.info(f"[Chain] Keys: {list(result.keys())}")

    # Try every possible key — never silently return empty
    answer = (result.get("answer") or result.get("result") or
              result.get("output_text") or result.get("text") or "")

    if not answer:
        logger.error(f"[Chain] Empty! Full result: {result}")
        answer = "Could not generate a response. Check server logs."

    sources = result.get("source_documents", [])
    logger.info(f"[Chain] Answer: {len(answer)} chars | Sources: {len(sources)}")
    return answer, sources