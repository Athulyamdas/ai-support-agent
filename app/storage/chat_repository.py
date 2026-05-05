"""
app/storage/chat_repository.py — All database read/write operations for chat.

WHAT IS THIS FILE?
──────────────────
This file is the "data access layer" — the only place in the project
that directly reads from or writes to the database.

WHY SEPARATE THIS FROM OTHER FILES?
─────────────────────────────────────
Imagine tomorrow you want to switch from SQLite to MySQL.
You only change THIS file — nothing else in the project needs to change.
This is called the Repository Pattern — a very common interview topic.

ANALOGY:
  database.py       = the filing cabinet (structure)
  chat_repository.py = the filing clerk (knows how to file and find things)
  conversation_chain.py = the manager (tells the clerk what to do)
"""

from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.storage.database import get_db_session, SessionModel, MessageModel
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_session(session_id: str) -> SessionModel:
    """
    Create a new session record in the database.

    Called when a brand new conversation starts.

    SQL equivalent:
        INSERT INTO sessions (session_id, created_at, message_count, status)
        VALUES (?, NOW(), 0, 'active')
    """
    db = get_db_session()
    try:
        session = SessionModel(session_id=session_id)
        db.add(session)       # stage the new record
        db.commit()           # write it to the file
        db.refresh(session)   # reload from DB to get auto-generated values
        logger.info(f"Created session in DB: {session_id}")
        return session
    finally:
        db.close()            # always close — releases the DB connection


def get_session(session_id: str) -> Optional[SessionModel]:
    """
    Find a session by its ID. Returns None if not found.

    SQL equivalent:
        SELECT * FROM sessions WHERE session_id = ? LIMIT 1
    """
    db = get_db_session()
    try:
        return db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
    finally:
        db.close()


def get_or_create_session(session_id: str) -> SessionModel:
    """
    Get session if it exists, create it if it doesn't.

    This is the main function called by the conversation chain.
    It handles both new and returning users cleanly.
    """
    session = get_session(session_id)
    if session is None:
        session = create_session(session_id)
    return session


def update_session_message_count(session_id: str, count: int) -> None:
    """
    Update the message count and last-updated timestamp for a session.

    SQL equivalent:
        UPDATE sessions
        SET message_count = ?, updated_at = NOW()
        WHERE session_id = ?
    """
    db = get_db_session()
    try:
        session = db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        if session:
            session.message_count = count
            session.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def list_all_sessions() -> List[SessionModel]:
    """
    Get all sessions ordered by most recent first.

    SQL equivalent:
        SELECT * FROM sessions ORDER BY updated_at DESC
    """
    db = get_db_session()
    try:
        return db.query(SessionModel).order_by(
            SessionModel.updated_at.desc()
        ).all()
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MESSAGE OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def save_message(session_id: str, role: str, content: str) -> MessageModel:
    """
    Save a single message to the database.

    Parameters:
        session_id : which conversation this message belongs to
        role       : 'human' (customer) or 'ai' (Aria)
        content    : the actual text of the message

    SQL equivalent:
        INSERT INTO messages (session_id, role, content, created_at)
        VALUES (?, ?, ?, NOW())
    """
    db = get_db_session()
    try:
        message = MessageModel(
            session_id=session_id,
            role=role,
            content=content,
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        logger.debug(f"Saved message [{role}] for session {session_id[:8]}...")
        return message
    finally:
        db.close()


def get_messages_for_session(session_id: str) -> List[MessageModel]:
    """
    Get all messages for a session, ordered oldest first.

    This is used to rebuild conversation history after a server restart.

    SQL equivalent:
        SELECT * FROM messages
        WHERE session_id = ?
        ORDER BY created_at ASC
    """
    db = get_db_session()
    try:
        return db.query(MessageModel).filter(
            MessageModel.session_id == session_id
        ).order_by(MessageModel.created_at.asc()).all()
    finally:
        db.close()


def messages_to_langchain(session_id: str) -> List[BaseMessage]:
    """
    Load messages from DB and convert to LangChain message objects.

    WHY DO WE NEED THIS?
    ─────────────────────
    The database stores messages as plain text rows.
    LangChain needs messages as HumanMessage / AIMessage objects.
    This function converts between the two formats.

    Database row:  { role: 'human', content: 'Hi I cant log in' }
    LangChain obj: HumanMessage(content='Hi I cant log in')

    Database row:  { role: 'ai', content: 'I am sorry to hear that...' }
    LangChain obj: AIMessage(content='I am sorry to hear that...')
    """
    db_messages = get_messages_for_session(session_id)
    langchain_messages = []

    for msg in db_messages:
        if msg.role == "human":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            langchain_messages.append(AIMessage(content=msg.content))

    return langchain_messages


def delete_session_messages(session_id: str) -> int:
    """
    Delete all messages for a session. Returns count of deleted messages.

    SQL equivalent:
        DELETE FROM messages WHERE session_id = ?
    """
    db = get_db_session()
    try:
        count = db.query(MessageModel).filter(
            MessageModel.session_id == session_id
        ).delete()
        db.commit()
        logger.info(f"Deleted {count} messages for session {session_id[:8]}...")
        return count
    finally:
        db.close()