"""
app/agent/conversation_chain.py — Multi-turn chain with SQLite persistence.

WHAT CHANGED FROM DAY 1?
─────────────────────────
Day 1: History stored in a Python dict in RAM → lost on restart
Day 2: History stored in SQLite on disk → survives restarts

The key change is in _get_history():
  Before: return _session_histories[session_id]  (RAM dict)
  After:  load from DB, return populated ChatMessageHistory object

Everything else (the LCEL chain, the chat() function) stays the same.
This is good design — the chain doesn't care WHERE history comes from.
"""

from __future__ import annotations
from typing import Dict

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

import config
from app.agent.llm_factory import build_llm
from app.storage.chat_repository import (
    get_or_create_session,
    save_message,
    messages_to_langchain,
    update_session_message_count,
    delete_session_messages,
    list_all_sessions,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Aria, a friendly and highly competent AI customer support agent.

Your responsibilities:
1. Answer product and account questions clearly and concisely.
2. Search the knowledge base when you need factual detail (Day 3 feature).
3. Look up customer account info via the CRM when relevant (Day 4 feature).
4. Escalate to a human agent if:
   - The issue involves billing disputes over $200.
   - The customer expresses strong frustration after repeated attempts.
   - The query is outside your scope (legal, medical, regulatory).

Tone guidelines:
- Warm, professional, never robotic.
- Acknowledge the customer's feelings before diving into solutions.
- Keep replies concise — 3-5 sentences unless detail is explicitly needed.
- Never fabricate policies, prices, or account data.

If you don't know something, say so clearly and offer to escalate.
"""

# ── In-memory cache (RAM) ─────────────────────────────────────────────────────
# WHY KEEP A RAM CACHE AT ALL?
# ─────────────────────────────
# Reading from disk (SQLite) on EVERY message would be slow.
# Strategy: load from DB once (first message of session), keep in RAM after.
# This is called a "write-through cache":
#   - Reads: RAM first, DB if not in RAM
#   - Writes: RAM AND DB simultaneously
_session_histories: Dict[str, ChatMessageHistory] = {}

# Lazily cached chain (built once, reused forever)
_chain_with_history = None


# ── Chain Builder ─────────────────────────────────────────────────────────────

def _get_chain():
    """Build the LCEL chain once and cache it."""
    global _chain_with_history
    if _chain_with_history is not None:
        return _chain_with_history

    llm = build_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    raw_chain = prompt | llm | StrOutputParser()

    _chain_with_history = RunnableWithMessageHistory(
        raw_chain,
        _get_history,                    # this function provides history
        input_messages_key="input",
        history_messages_key="history",
    )
    return _chain_with_history


# ── History Provider ──────────────────────────────────────────────────────────

def _get_history(session_id: str) -> BaseChatMessageHistory:
    """
    Return the ChatMessageHistory for a session.

    FLOW:
    ─────
    1. Check RAM cache — if found, return immediately (fast path)
    2. If not in RAM, load from SQLite database (cold start)
    3. Store in RAM cache for future messages in this session

    This means:
    - First message of a session: loads from DB (slightly slower)
    - All subsequent messages: served from RAM (fast)
    - Server restart: RAM is empty, so first message loads from DB again
      BUT the history is not lost — it's all in SQLite!
    """
    if session_id in _session_histories:
        return _session_histories[session_id]   # RAM cache hit

    # Cold start — load history from database
    logger.info(f"Loading session from DB: {session_id[:8]}...")

    # Ensure the session row exists in the sessions table
    get_or_create_session(session_id)

    # Load all past messages and convert to LangChain format
    history = ChatMessageHistory()
    past_messages = messages_to_langchain(session_id)

    if past_messages:
        history.messages.extend(past_messages)
        logger.info(f"Restored {len(past_messages)} messages for session {session_id[:8]}...")
    else:
        logger.info(f"New session started: {session_id[:8]}...")

    # Store in RAM cache
    _session_histories[session_id] = history
    return history


# ── Public Functions ──────────────────────────────────────────────────────────

def chat(session_id: str, user_message: str) -> dict:
    """
    Send a message and get the agent's reply.
    Saves both the human message and AI reply to SQLite.

    Returns dict with: session_id, reply, turn, history_len
    """
    chain = _get_chain()
    logger.debug(f"[{session_id[:8]}] USER → {user_message!r}")

    # Save human message to DB BEFORE calling LLM
    save_message(session_id=session_id, role="human", content=user_message)

    # Invoke the chain (this also updates the RAM history automatically)
    reply: str = chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )

    # Save AI reply to DB AFTER getting response
    save_message(session_id=session_id, role="ai", content=reply)

    # Update session stats
    history = _get_history(session_id)
    history_len = len(history.messages)
    turn = history_len // 2
    update_session_message_count(session_id, history_len)

    logger.debug(f"[{session_id[:8]}] ARIA → {reply[:80]}...")
    logger.info(f"[{session_id[:8]}] turn={turn}, history_len={history_len}")

    return {
        "session_id": session_id,
        "reply": reply,
        "turn": turn,
        "history_len": history_len,
    }


def clear_session(session_id: str) -> bool:
    """
    Clear session from both RAM and database.
    Returns True if session existed.
    """
    in_ram = session_id in _session_histories
    _session_histories.pop(session_id, None)   # remove from RAM

    deleted = delete_session_messages(session_id)   # remove from DB
    existed = in_ram or deleted > 0

    logger.info(f"Cleared session: {session_id[:8]}... (existed={existed})")
    return existed


def list_sessions() -> list[str]:
    """Return all session IDs from the database (not just RAM)."""
    sessions = list_all_sessions()
    return [s.session_id for s in sessions]