"""
app/agent/conversation_chain.py — Multi-turn chain with SQLite + KB + CRM.

WHAT CHANGED FROM DAY 3?
─────────────────────────
Day 3: Aria searches KB for policy information
Day 4: Aria ALSO looks up real customer account data from CRM

THE NEW FLOW:
─────────────
User message
    │
    ├─► 1. Search FAISS KB for relevant policy chunks
    │
    ├─► 2. Scan message for customer identifier (email/phone)
    │         └─► If found: look up customer in CRM
    │
    └─► 3. Build prompt with:
              System + KB context + CRM context + History + Message
                  │
                  └─► LLM generates personalised, accurate response

SESSION CRM MEMORY:
───────────────────
Once a customer is identified in a session, we remember who they are
for the rest of the conversation. They don't need to repeat their email
every message. This is stored in _session_customers dict.
"""

from __future__ import annotations
from typing import Dict, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from app.agent.llm_factory import build_llm
from app.agent.knowledge_base import search_knowledge_base
from app.api.crm_service import (
    lookup_customer_from_message,
    format_customer_context,
)
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
SYSTEM_PROMPT = """You are Aria, a friendly and highly competent AI customer support agent for a bank.

KNOWLEDGE BASE CONTEXT (Banking Policies):
{kb_context}

CUSTOMER ACCOUNT INFORMATION:
{crm_context}

Your responsibilities:
1. Use the KNOWLEDGE BASE CONTEXT to answer policy questions accurately.
2. Use the CUSTOMER ACCOUNT INFORMATION to personalise responses.
   - Address the customer by their first name.
   - Reference their actual account details (status, balance, cards, loans).
   - If account is frozen/suspended, explain why and what steps to take.
   - If loan is overdue, mention the overdue amount and urgency.
3. If no customer information is available, ask for their registered email
   or phone number to look up their account.
4. Escalate to a human agent if:
   - Disputed transaction over Rs. 10,000.
   - Customer expresses strong frustration after repeated attempts.
   - Issue requires branch visit or legal/regulatory guidance.

Tone guidelines:
- Warm, professional, never robotic.
- Always use the customer's first name when you know it.
- Acknowledge feelings before solutions.
- Keep replies concise — 3-5 sentences unless more detail is needed.
- Never fabricate account balances, transaction data, or policy details.

If you cannot find a customer's account, say so politely and ask them to
visit the nearest branch or call 1800-XXX-XXXX.
"""

# ── In-memory caches ──────────────────────────────────────────────────────────
_session_histories: Dict[str, ChatMessageHistory] = {}

# NEW: Remember which customer is associated with each session
# Once identified, customer persists for the whole conversation
_session_customers: Dict[str, Optional[dict]] = {}

_chain_with_history = None


def _get_chain():
    """Build and cache the LCEL chain."""
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
        logger.info(
            f"Restored {len(past_messages)} messages for session {session_id[:8]}..."
        )
    else:
        logger.info(f"New session started: {session_id[:8]}...")

    _session_histories[session_id] = history
    return history


def _build_kb_context(query: str) -> str:
    """Search KB and return formatted context string."""
    try:
        results = search_knowledge_base(query, top_k=3)
        if not results:
            return "No relevant policy information found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Source {i}: {r['source']}]\n{r['content']}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning(f"KB search failed: {e}")
        return "Knowledge base unavailable."


def _get_crm_context(session_id: str, user_message: str) -> tuple[str, Optional[dict]]:
    """
    Get CRM context for this message.

    LOGIC:
    ──────
    1. Check if customer already identified in this session → reuse
    2. If not, scan current message for email/phone
    3. If found → look up CRM → cache for session
    4. Return formatted context string + customer dict

    This means:
    - Turn 1: "My email is alice@example.com" → finds Alice, caches her
    - Turn 2: "What's my balance?" → already knows it's Alice, no re-lookup
    - Turn 3: "Do I have any loans?" → still Alice, answers from her profile
    """
    # Check session cache first
    if session_id in _session_customers:
        customer = _session_customers[session_id]
        if customer:
            logger.debug(
                f"[{session_id[:8]}] Using cached CRM customer: {customer['name']}"
            )
            return format_customer_context(customer), customer
        else:
            # Previously checked and found no customer
            return _no_customer_context(), None

    # Try to identify customer from current message
    customer = lookup_customer_from_message(user_message)

    # Cache result (even if None — avoids re-checking every message)
    _session_customers[session_id] = customer

    if customer:
        logger.info(
            f"[{session_id[:8]}] Customer identified: {customer['name']} "
            f"({customer['customer_id']})"
        )
        return format_customer_context(customer), customer
    else:
        return _no_customer_context(), None


def _no_customer_context() -> str:
    """Context string when no customer has been identified yet."""
    return (
        "No customer account identified yet.\n"
        "If the customer mentions account-specific details, ask for their "
        "registered email address or 10-digit phone number to look up their account."
    )


# ── Public entry point ────────────────────────────────────────────────────────

def chat(session_id: str, user_message: str) -> dict:
    """
    Send a message and get the agent's reply.
    Now includes KB search + CRM lookup before calling the LLM.

    Returns dict with:
        session_id, reply, turn, history_len,
        customer_identified, customer_name
    """
    chain = _get_chain()
    logger.debug(f"[{session_id[:8]}] USER → {user_message!r}")

    # Step 1: Search knowledge base
    kb_context = _build_kb_context(user_message)

    # Step 2: Look up customer in CRM
    crm_context, customer = _get_crm_context(session_id, user_message)

    # Step 3: Save human message to DB
    save_message(session_id=session_id, role="human", content=user_message)

    # Step 4: Invoke LLM chain with both contexts
    reply: str = chain.invoke(
        {
            "input": user_message,
            "kb_context": kb_context,
            "crm_context": crm_context,
        },
        config={"configurable": {"session_id": session_id}},
    )

    # Step 5: Save AI reply to DB
    save_message(session_id=session_id, role="ai", content=reply)

    # Step 6: Update session stats
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
        "customer_identified": customer is not None,
        "customer_name": customer["name"] if customer else None,
    }


def clear_session(session_id: str) -> bool:
    """Clear session from RAM cache and database."""
    in_ram = session_id in _session_histories
    _session_histories.pop(session_id, None)
    _session_customers.pop(session_id, None)   # also clear CRM cache
    deleted = delete_session_messages(session_id)
    existed = in_ram or deleted > 0
    logger.info(f"Cleared session: {session_id[:8]}... (existed={existed})")
    return existed


def list_sessions() -> list[str]:
    """Return all session IDs from the database."""
    return [s.session_id for s in list_all_sessions()]