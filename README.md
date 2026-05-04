# AI Customer Support Agent

A portfolio project demonstrating an **Agentic AI Developer** skill set.

## Architecture

```
Chat UI  ──►  Flask REST API  ──►  LangChain Agent  ──►  Storage Layer
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                      Answer        Search KB      Escalate
                      directly      (FAISS)       to human
                          │             │
                          └──────┬──────┘
                                 ▼
                        CRM lookup (SQLite / JSON)
```

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini **or** Anthropic Claude (swap via `.env`) |
| Agent framework | LangChain |
| Knowledge base | FAISS vector store |
| Chat history | SQLite via SQLAlchemy |
| CRM | Mock JSON file |
| API | Flask + Flask-CORS |

## Day-by-Day Plan

| Day | Goal | Status |
|-----|------|--------|
| 1 | Project setup, LLM connection, multi-turn chain | ✅ |
| 2 | SQLite persistence, session management | 🔜 |
| 3 | FAISS knowledge base ingestion & retrieval | 🔜 |
| 4 | CRM integration, LangChain Tools | 🔜 |
| 5 | Decision router (answer/search/escalate) | 🔜 |
| 6 | Full LangChain Agent with tool-use | 🔜 |
| 7 | Polish, error handling, portfolio write-up | 🔜 |

## Quick Start

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or ANTHROPIC_API_KEY

# 3. Run interactive CLI test (no server needed)
python scripts/smoke_test.py

# 4. Start the Flask API
python main.py

# 5. Test the API
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I cannot log into my account"}'
```

## Project Structure

```
ai-support-agent/
├── main.py                    # Entry point
├── config.py                  # Centralised env config
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py            # Flask app factory
│   ├── agent/
│   │   ├── llm_factory.py     # Build LLM from config
│   │   └── conversation_chain.py  # Multi-turn chain + memory
│   ├── api/
│   │   └── routes.py          # REST endpoints
│   └── utils/
│       └── logger.py          # Colourised logger
├── data/
│   ├── faiss_index/           # (populated Day 3)
│   └── mock_crm/
│       └── customers.json
├── scripts/
│   └── smoke_test.py          # CLI end-to-end test
└── tests/
    └── test_day1.py           # Pytest suite
```

## API Reference (Day 1)

### `POST /api/chat`
```json
// Request
{ "message": "I can't log in", "session_id": "optional-uuid" }

// Response
{
  "session_id": "abc-123",
  "reply": "I'm sorry to hear that...",
  "turn": 1,
  "history_len": 2
}
```

### `DELETE /api/session/<id>`
Clear a session's memory buffer.

### `GET /api/health`
Liveness check.
