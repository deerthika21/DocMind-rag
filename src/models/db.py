import json
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from config import SQLITE_URL

Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "sessions"
    id         = Column(String,   primary_key=True, default=lambda: str(uuid.uuid4()))
    title      = Column(String,   default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    messages   = relationship("ChatMessage", back_populates="session",
                              cascade="all, delete-orphan", order_by="ChatMessage.id")


class ChatMessage(Base):
    __tablename__ = "messages"
    id         = Column(Integer,  primary_key=True, autoincrement=True)
    session_id = Column(String,   ForeignKey("sessions.id"), nullable=False)
    role       = Column(String,   nullable=False)
    content    = Column(Text,     nullable=False)
    sources    = Column(Text,     default="[]")
    created_at = Column(DateTime, default=datetime.utcnow)
    session    = relationship("ChatSession", back_populates="messages")


async_engine      = create_async_engine(SQLITE_URL, echo=False)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession,
                                  expire_on_commit=False)


async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


def parse_sources(raw: str) -> list:
    try:
        return json.loads(raw or "[]")
    except Exception:
        return []