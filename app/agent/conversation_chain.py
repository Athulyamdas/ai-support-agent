"""
app/agent/conversation_chain.py — Multi-turn chain with SQLite + KB search.

WHAT CHANGED FROM DAY 2?
─────────────────────────
Day 2: Aria answers from LLM knowledge only (can make things up)
Day 3: Aria SEARCHES the banking knowledge base first, then answers
       using real information from your documents.

THE NEW FLOW:
─────────────
User message
    │
    ├─► Search FAISS knowledge base for relevant chunks
    │       │
    │       └─► Top 3 relevant banking policy chunks
    │
    └─► Build prompt with: System + KB context + History + User message
            │
            └─► LLM generates answer GROUNDED in real KB content
"""

from __future__ import annotations
from typing import Dict

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory

import config
from app.agent.llm_factory import build_llm
from app.agent.knowledge_base import search_knowledge_base
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
# Notice the new {context} placeholder — this is where KB results get injected
SYSTEM_PROMPT = """You are Aria, a friendly and highly competent AI customer support agent for a bank.

KNOWLEDGE BASE CONTEXT:
The following information has been retrieved from our banking policy documents.
Use this information to answer the customer's question accurately.
If the context does not contain the answer, say so honestly — do not make up information.

{context}

Your responsibilities:
1. Answer using the knowledge base context above whenever relevant.
2. For account-specific queries (balance, transactions), tell the customer you will look up their account (Day 4 feature).
3. Escalate to a human agent if:
   - The issue involves a disputed transaction over Rs. 10,000.
   - The customer expresses strong frustration after repeated attempts.
   - The query requires legal or regulatory guidance.

Tone guidelines:
- Warm, professional, never robotic.
- Acknowledge the customer's feelings before diving into solutions.
- Keep replies concise — 3-5 sentences unless more detail is needed.
- Never fabricate policies, rates, or account data.
- Always mention relevant charges, deadlines, or required documents clearly.

If you don't know something, say so clearly and offer to escalate.
"""

# ── Session memory cache ──────────────────────────────────────────────────────
_session_histories: Dict[str, ChatMessageHistory] = {}
_chain_with_history = None


def _get_chain():
    """Build the LCEL chain with KB context injection."""
    global _chain_with_history
    if _chain_with_history is not None:
        return _chain_with_history

    llm = build_llm()

    # NEW: Prompt now has a {context} variable for KB results
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    raw_chain = prompt | llm | StrOutputParser()

    _chain_with_history = RunnableWithMessageHistory(
        raw_chain,
        _get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return _chain_with_history


def _get_history(session_id: str) -> BaseChatMessageHistory:
    """Load history from DB on first access, then cache in RAM."""
    if session_id in _session_histories:
        return _session_histories[session_id]

    logger.info(f"Loading session from DB: {session_id[:8]}...")
    get_or_create_session(session_id)

    history = ChatMessageHistory()
    past_messages = messages_to_langchain(session_id)

    if past_messages:
        history.messages.extend(past_messages)
        logger.info(f"Restored {len(past_messages)} messages for session {session_id[:8]}...")
    else:
        logger.info(f"New session started: {session_id[:8]}...")

    _session_histories[session_id] = history
    return history


def _build_kb_context(query: str) -> str:
    """
    Search the knowledge base and format results as context string.

    This context is injected into the system prompt so the LLM
    can answer based on real banking policy documents.

    If KB search fails (e.g. index not built), returns empty context
    gracefully — the LLM will still work, just without KB grounding.
    """
    try:
        results = search_knowledge_base(query, top_k=3)

        if not results:
            return "No relevant information found in knowledge base."

        # Format results into a readable context block
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result['source']}]\n{result['content']}"
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        logger.warning(f"KB search failed: {e}. Continuing without KB context.")
        return "Knowledge base unavailable. Answer based on general banking knowledge."


def chat(session_id: str, user_message: str) -> dict:
    """
    Send a message and get the agent's reply.
    Now includes KB search before calling the LLM.

    Returns dict with: session_id, reply, turn, history_len, kb_context_used
    """
    chain = _get_chain()
    logger.debug(f"[{session_id[:8]}] USER → {user_message!r}")

    # Step 1: Search knowledge base for relevant context
    kb_context = _build_kb_context(user_message)
    logger.debug(f"[{session_id[:8]}] KB context length: {len(kb_context)} chars")

    # Step 2: Save human message to DB
    save_message(session_id=session_id, role="human", content=user_message)

    # Step 3: Invoke chain with KB context injected into system prompt
    reply: str = chain.invoke(
        {
            "input": user_message,
            "context": kb_context,    # ← injected into {context} in system prompt
        },
        config={"configurable": {"session_id": session_id}},
    )

    # Step 4: Save AI reply to DB
    save_message(session_id=session_id, role="ai", content=reply)

    # Step 5: Update session stats
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
        "kb_searched": True,
    }


def clear_session(session_id: str) -> bool:
    """Clear session from RAM and database."""
    in_ram = session_id in _session_histories
    _session_histories.pop(session_id, None)
    deleted = delete_session_messages(session_id)
    existed = in_ram or deleted > 0
    logger.info(f"Cleared session: {session_id[:8]}... (existed={existed})")
    return existed


def list_sessions() -> list[str]:
    """Return all session IDs from the database."""
    sessions = list_all_sessions()
    return [s.session_id for s in sessions]