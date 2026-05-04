"""
app/agent/llm_factory.py — Build the correct LangChain LLM from config.
Supports OpenAI, Groq, and Anthropic — swap via LLM_PROVIDER in .env
"""

from langchain_core.language_models import BaseChatModel
import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_llm() -> BaseChatModel:
    provider = config.LLM_PROVIDER
    logger.info(f"Building LLM: provider={provider}, model={config.LLM_MODEL}")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            openai_api_key=config.OPENAI_API_KEY,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            groq_api_key=config.GROQ_API_KEY,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Supported values: 'openai', 'groq', 'anthropic'."
    )