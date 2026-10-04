from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

sessions = {}


async def create_session(db=None, title="New Chat"):
    """Create a new chat session"""
    session_id = str(uuid4())
    sessions[session_id] = {
        "id": session_id,
        "title": title,
        "messages": [],
        "updated_at": __import__('datetime').datetime.utcnow().isoformat()
    }
    logger.info(f"[History] Created session {session_id}")
    return sessions[session_id]


async def get_session(session_id, db=None):
    """Get session by ID"""
    return sessions.get(session_id)


async def list_sessions(db=None):
    """List all sessions"""
    return list(sessions.values())


async def delete_session(session_id, db=None):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"[History] Deleted session {session_id}")
        return True
    return False


async def add_message(session_id, role, content, sources=None, db=None):
    """Add message to session"""
    if session_id in sessions:
        sessions[session_id]["messages"].append({
            "role": role,
            "content": content,
            "sources": sources or []
        })
        sessions[session_id]["updated_at"] = __import__('datetime').datetime.utcnow().isoformat()
        logger.info(f"[History] Added {role} message to {session_id}")


async def get_messages(session_id, db=None):
    """Get all messages in session"""
    if session_id in sessions:
        return sessions[session_id]["messages"]
    return []