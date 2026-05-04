"""
config.py — Centralised configuration loader.

All env vars are read once here; the rest of the app imports from this module.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of where you run the app)
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# ── Storage ───────────────────────────────────────────────────────────────────
SQLITE_DB_PATH: Path = ROOT_DIR / os.getenv("SQLITE_DB_PATH", "data/chat_history.db")
FAISS_INDEX_PATH: Path = ROOT_DIR / os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
MOCK_CRM_PATH: Path = ROOT_DIR / os.getenv("MOCK_CRM_PATH", "data/mock_crm/customers.json")

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
ESCALATION_THRESHOLD: int = int(os.getenv("ESCALATION_THRESHOLD", "3"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


def validate() -> None:
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example → .env and add your key."
        )
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example → .env and add your key."
        )
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example → .env and add your key."
        )