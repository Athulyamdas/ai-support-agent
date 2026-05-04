"""
app/api/routes.py — Flask REST API (Day 1: /chat, /session, /health).

All endpoints return JSON. Error responses always include an 'error' key.

Day 1 endpoints
───────────────
POST  /api/chat               Send a message, get a reply
DELETE /api/session/<id>      Clear a session's memory
GET   /api/sessions           List active session IDs  [debug]
GET   /api/health             Liveness check
"""

from __future__ import annotations

import uuid
from flask import Blueprint, request, jsonify, Response

from app.agent.conversation_chain import chat, clear_session, list_sessions
from app.utils.logger import get_logger

logger = get_logger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")


# ── Helper ────────────────────────────────────────────────────────────────────

def _error(message: str, status: int = 400) -> tuple[Response, int]:
    logger.warning(f"API error {status}: {message}")
    return jsonify({"error": message}), status


# ── Routes ────────────────────────────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health() -> tuple[Response, int]:
    """Quick liveness check — no LLM call."""
    return jsonify({"status": "ok", "service": "AI Support Agent"}), 200


@api_bp.route("/chat", methods=["POST"])
def chat_endpoint() -> tuple[Response, int]:
    """
    Send a message to the support agent.

    Request body (JSON)
    ───────────────────
    {
        "message"    : "I can't log into my account",   // required
        "session_id" : "abc-123"                         // optional; created if absent
    }

    Response
    ────────
    {
        "session_id"  : "abc-123",
        "reply"       : "I'm sorry to hear that...",
        "turn"        : 1,
        "history_len" : 2
    }
    """
    body = request.get_json(silent=True)

    if not body:
        return _error("Request body must be JSON with a 'message' field.")

    message: str = body.get("message", "").strip()
    if not message:
        return _error("'message' field is required and cannot be empty.")

    # Auto-generate session_id if client doesn't supply one
    session_id: str = body.get("session_id") or str(uuid.uuid4())

    try:
        result = chat(session_id=session_id, user_message=message)
        return jsonify(result), 200
    except Exception as exc:
        logger.exception(f"Unhandled error during chat for session {session_id}")
        return _error(f"Internal agent error: {exc}", status=500)


@api_bp.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str) -> tuple[Response, int]:
    """
    Clear a session's conversation memory.

    Useful for 'Start new conversation' UI buttons.
    """
    existed = clear_session(session_id)
    return jsonify({
        "session_id": session_id,
        "cleared": existed,
        "message": "Session cleared." if existed else "Session not found.",
    }), 200


@api_bp.route("/sessions", methods=["GET"])
def get_sessions() -> tuple[Response, int]:
    """List all active session IDs. For development/debugging only."""
    sessions = list_sessions()
    return jsonify({"active_sessions": sessions, "count": len(sessions)}), 200
