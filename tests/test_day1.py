"""
tests/test_day1.py — Unit tests for Day 1 components.

Run: pytest tests/test_day1.py -v

These tests mock the LLM so they run without a real API key.
"""

from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock


# ── Config tests ──────────────────────────────────────────────────────────────

def test_config_loads():
    import config
    assert hasattr(config, "LLM_PROVIDER")
    assert hasattr(config, "MAX_CONVERSATION_HISTORY")


def test_config_validate_raises_without_key(monkeypatch):
    import config
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        config.validate()


# ── LLM factory tests ─────────────────────────────────────────────────────────

def test_build_llm_unknown_provider(monkeypatch):
    import config
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    from app.agent.llm_factory import build_llm
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm()


# ── Conversation chain tests ──────────────────────────────────────────────────

@patch("app.agent.conversation_chain.build_llm")
def test_chat_returns_expected_keys(mock_build_llm):
    """chat() should return session_id, reply, turn, history_len."""
    # Mock the LLM so no real API call is made
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Hello! How can I help?")
    mock_build_llm.return_value = mock_llm

    # Also mock the ConversationChain invoke
    with patch("app.agent.conversation_chain.ConversationChain") as mock_chain_cls:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"response": "Hello! How can I help?"}
        mock_chain.memory.chat_memory.messages = ["msg1", "msg2"]
        mock_chain_cls.return_value = mock_chain

        from app.agent.conversation_chain import chat, clear_session
        result = chat(session_id="test-123", user_message="Hi there")

        assert result["session_id"] == "test-123"
        assert result["reply"] == "Hello! How can I help?"
        assert "turn" in result
        assert "history_len" in result

        clear_session("test-123")


@patch("app.agent.conversation_chain.build_llm")
def test_clear_session(mock_build_llm):
    """clear_session returns True when session existed, False otherwise."""
    mock_llm = MagicMock()
    mock_build_llm.return_value = mock_llm

    with patch("app.agent.conversation_chain.ConversationChain") as mock_chain_cls:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {"response": "Test reply"}
        mock_chain.memory.chat_memory.messages = []
        mock_chain_cls.return_value = mock_chain

        from app.agent.conversation_chain import chat, clear_session
        chat(session_id="sess-to-clear", user_message="test")

        assert clear_session("sess-to-clear") is True
        assert clear_session("sess-to-clear") is False   # already gone


# ── Flask API tests ───────────────────────────────────────────────────────────

@pytest.fixture
def flask_client():
    """Create a test Flask client with mocked config validation."""
    with patch("config.validate"):
        from app import create_app
        app = create_app()
        app.testing = True
        with app.test_client() as client:
            yield client


def test_health_endpoint(flask_client):
    resp = flask_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


def test_chat_missing_message(flask_client):
    resp = flask_client.post("/api/chat", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_chat_empty_message(flask_client):
    resp = flask_client.post("/api/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_sessions_endpoint(flask_client):
    resp = flask_client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "active_sessions" in data
    assert "count" in data
