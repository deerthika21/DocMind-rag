from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer:     str
    sources:    list[str]
    session_id: str


class UploadResponse(BaseModel):
    message:        str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status:      str
    chain_ready: bool
    docs_indexed: int


class SessionOut(BaseModel):
    id:         str
    title:      str
    updated_at: str


class MessageOut(BaseModel):
    role:    str
    content: str
    sources: list[str]


class SessionDetail(BaseModel):
    id:       str
    title:    str
    messages: list[MessageOut]