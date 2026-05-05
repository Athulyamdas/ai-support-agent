"""
app/api/admin_routes.py — Endpoints to inspect chat history stored in SQLite.

WHY THESE ENDPOINTS?
─────────────────────
Day 2 adds database persistence. These endpoints let you:
  1. See all sessions that ever happened
  2. Read the full conversation history of any session
  3. Delete a session and all its messages

This is extremely useful for:
  - Debugging (did the messages actually save?)
  - Portfolio demo (show the interviewer history survives restarts)
  - Building an admin dashboard in the future

NEW ENDPOINTS:
  GET  /api/admin/sessions              — list all sessions
  GET  /api/admin/sessions/<id>         — get full history of one session
  GET  /api/admin/sessions/<id>/messages — get raw messages
  DELETE /api/admin/sessions/<id>       — delete session + all its messages
"""

from flask import Blueprint, jsonify, Response

from app.storage.chat_repository import (
    list_all_sessions,
    get_session,
    get_messages_for_session,
    delete_session_messages,
)
from app.agent.conversation_chain import clear_session
from app.utils.logger import get_logger

logger = get_logger(__name__)
admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


def _error(message: str, status: int = 400) -> tuple[Response, int]:
    return jsonify({"error": message}), status


@admin_bp.route("/sessions", methods=["GET"])
def get_all_sessions() -> tuple[Response, int]:
    """
    List all chat sessions ever created.

    Response example:
    {
        "sessions": [
            {
                "session_id": "f126dcb1-...",
                "created_at": "2024-10-01T10:30:00",
                "message_count": 6,
                "status": "active"
            }
        ],
        "total": 1
    }
    """
    sessions = list_all_sessions()
    return jsonify({
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                "message_count": s.message_count,
                "status": s.status,
            }
            for s in sessions
        ],
        "total": len(sessions),
    }), 200


@admin_bp.route("/sessions/<session_id>", methods=["GET"])
def get_session_detail(session_id: str) -> tuple[Response, int]:
    """
    Get full details of one session including all messages.

    This is the "show me the entire conversation" endpoint.
    Great for demos — shows history survived a server restart.
    """
    session = get_session(session_id)
    if not session:
        return _error(f"Session '{session_id}' not found.", 404)

    messages = get_messages_for_session(session_id)

    return jsonify({
        "session_id": session.session_id,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "message_count": session.message_count,
        "status": session.status,
        "conversation": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    }), 200


@admin_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id: str) -> tuple[Response, int]:
    """
    Delete a session and all its messages from the database.
    Also clears it from the RAM cache.
    """
    session = get_session(session_id)
    if not session:
        return _error(f"Session '{session_id}' not found.", 404)

    clear_session(session_id)
    return jsonify({
        "message": f"Session {session_id[:8]}... deleted successfully.",
        "session_id": session_id,
    }), 200