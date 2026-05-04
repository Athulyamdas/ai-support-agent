"""
app/agent/conversation_chain.py — Day 1 core: multi-turn chain with memory.

Uses the modern LangChain LCEL (LangChain Expression Language) pattern,
which is the recommended approach in LangChain >= 0.2 / 1.x.

Architecture
────────────
  User message
       │
       ▼
  LCEL chain: prompt | llm | parser
       │  ├─ SystemPrompt  (support persona + guardrails)
       │  ├─ ChatMessageHistory  (per-session message list)
       │  └─ LLM  (OpenAI / Anthropic, via llm_factory)
       │
       ▼
  Agent reply  ──►  (Day 2+) decision router → KB search / CRM / escalate
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
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are Aria, a friendly and highly competent AI customer support agent.

Your responsibilities:
1. Answer product and account questions clearly and concisely.
2. Search the knowledge base when you need factual detail (Day 2 feature).
3. Look up customer account info via the CRM when relevant (Day 2 feature).
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

# Per-session message histories (in-memory for Day 1)
_session_histories: Dict[str, ChatMessageHistory] = {}

# Lazily cached chain
_chain_with_history = None


def _get_chain():
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

    def _get_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in _session_histories:
            logger.info(f"Creating new session: {session_id}")
            _session_histories[session_id] = ChatMessageHistory()
        return _session_histories[session_id]

    _chain_with_history = RunnableWithMessageHistory(
        raw_chain,
        _get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return _chain_with_history


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = ChatMessageHistory()
    return _session_histories[session_id]


def clear_session(session_id: str) -> bool:
    existed = session_id in _session_histories
    _session_histories.pop(session_id, None)
    logger.info(f"Cleared session: {session_id} (existed={existed})")
    return existed


def list_sessions() -> list[str]:
    return list(_session_histories.keys())


def chat(session_id: str, user_message: str) -> dict:
    """
    Send a message and get the agent's reply.

    Returns dict with: session_id, reply, turn, history_len
    """
    chain = _get_chain()
    logger.debug(f"[{session_id}] USER → {user_message!r}")

    reply: str = chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )

    history = get_session_history(session_id)
    history_len = len(history.messages)
    turn = history_len // 2

    logger.debug(f"[{session_id}] ARIA → {reply!r}")
    logger.info(f"[{session_id}] turn={turn}, history_len={history_len}")

    return {
        "session_id": session_id,
        "reply": reply,
        "turn": turn,
        "history_len": history_len,
    }
