"""
app/agent/knowledge_base.py — FAISS vector store builder and search engine.

WHAT IS THIS FILE?
──────────────────
This file gives Aria the ability to SEARCH through banking documents
and find relevant information to answer customer questions accurately.

THE CORE CONCEPTS:
──────────────────

1. WHAT IS A VECTOR?
   A vector is a list of numbers that represents meaning.
   Example: "home loan documents" → [0.23, -0.87, 0.45, 0.12, ...]
   Two sentences with similar meaning will have similar vectors.
   This is how semantic search works — finding meaning, not just keywords.

2. WHAT IS AN EMBEDDING?
   An embedding model converts text into vectors.
   We use HuggingFace's free "all-MiniLM-L6-v2" model.
   It runs locally on your machine — no API call needed, completely free.

3. WHAT IS FAISS?
   FAISS (Facebook AI Similarity Search) is a library that:
   - Stores thousands of vectors efficiently
   - Finds the most similar vectors to a query vector VERY FAST
   - Runs entirely on your local machine
   Think of it as a "smart search engine" for your documents.

4. WHAT IS TEXT CHUNKING?
   You cannot feed an entire 10-page document to the LLM at once.
   So you split it into small overlapping chunks (paragraphs).
   Each chunk gets its own vector.
   When a user asks a question, FAISS finds the most relevant chunks.

FLOW:
─────
Build time (runs once):
  Text files → Split into chunks → Convert to vectors → Store in FAISS index

Query time (every message):
  User question → Convert to vector → Find similar chunks → Return top 3

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
'''from langchain_community.embeddings import HuggingFaceEmbeddings'''
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Embedding model ───────────────────────────────────────────────────────────
# This model runs LOCALLY — no API key, no cost, no internet needed.
# "all-MiniLM-L6-v2" is small (80MB), fast, and good quality.
# It will be downloaded once to your machine on first run.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Text splitter settings ────────────────────────────────────────────────────
# chunk_size: maximum characters per chunk
# chunk_overlap: how many characters to repeat between chunks
#   (overlap ensures context is not lost at chunk boundaries)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ── Cached vector store (loaded once, reused) ─────────────────────────────────
_vector_store: FAISS | None = None
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and cache it."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        logger.info("(First run downloads ~80MB — subsequent runs are instant)")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},   # use CPU (no GPU needed)
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully")
    return _embeddings


def _load_documents() -> List[Document]:
    """
    Read all .txt files from the knowledge_base folder.

    Each file becomes a Document object with:
      - page_content: the full text
      - metadata: source filename (useful for citing sources)
    """
    kb_path = Path(config.KNOWLEDGE_BASE_PATH)

    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base folder not found: {kb_path}\n"
            "Make sure data/knowledge_base/ exists with .txt files."
        )

    documents = []
    txt_files = list(kb_path.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in {kb_path}")

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")
        doc = Document(
            page_content=text,
            metadata={"source": file_path.name}
        )
        documents.append(doc)
        logger.info(f"Loaded document: {file_path.name} ({len(text)} chars)")

    return documents


def _split_documents(documents: List[Document]) -> List[Document]:
    """
    Split large documents into smaller overlapping chunks.

    WHY OVERLAP?
    If a chunk ends mid-sentence and the next starts fresh, you lose
    context. Overlap ensures important context spans across chunks.

    Example with chunk_size=20, overlap=5:
      Text:    "The home loan rate is 8.5% for good credit scores"
      Chunk 1: "The home loan rate is"
      Chunk 2: "rate is 8.5% for good"   ← "rate is" repeated for context
      Chunk 3: "for good credit scores"
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],  # try to split at natural boundaries
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def build_knowledge_base() -> FAISS:
    """
    Build the FAISS index from scratch and save it to disk.

    This function:
    1. Reads all .txt files from data/knowledge_base/
    2. Splits them into chunks
    3. Converts each chunk to a vector using the embedding model
    4. Stores all vectors in a FAISS index
    5. Saves the index to data/faiss_index/ for reuse

    Run this once when documents change. The saved index
    is loaded instantly on subsequent runs.
    """
    logger.info("Building knowledge base from scratch...")

    documents = _load_documents()
    chunks = _split_documents(documents)
    embeddings = _get_embeddings()

    logger.info(f"Converting {len(chunks)} chunks to vectors...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save to disk so we don't rebuild every time
    save_path = str(config.FAISS_INDEX_PATH)
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)

    logger.info(f"Knowledge base saved to: {save_path}")
    logger.info(f"Total chunks indexed: {len(chunks)}")

    return vector_store


def load_knowledge_base() -> FAISS:
    """
    Load the FAISS index from disk.
    Raises FileNotFoundError if index hasn't been built yet.
    """
    index_path = str(config.FAISS_INDEX_PATH)

    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}.\n"
            "Run: python scripts/build_kb.py first."
        )

    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True  # required flag in newer LangChain
    )
    logger.info(f"Knowledge base loaded from: {index_path}")
    return vector_store


def get_knowledge_base() -> FAISS:
    """
    Get the vector store — load from disk if available, build if not.
    Uses module-level cache so disk is only read once per server run.
    """
    global _vector_store

    if _vector_store is not None:
        return _vector_store   # already in memory

    index_path = os.path.join(str(config.FAISS_INDEX_PATH), "index.faiss")

    if os.path.exists(index_path):
        _vector_store = load_knowledge_base()
    else:
        logger.warning("FAISS index not found — building now (takes ~30 seconds)...")
        _vector_store = build_knowledge_base()

    return _vector_store


def search_knowledge_base(query: str, top_k: int = 3) -> List[dict]:
    """
    Search the knowledge base for chunks relevant to the query.

    Parameters
    ----------
    query  : The customer's question or keywords to search for
    top_k  : How many chunks to return (default 3)

    Returns
    -------
    List of dicts, each with:
        content  : the text chunk
        source   : which file it came from
        score    : similarity score (lower = more similar in FAISS)

    Example:
        search_knowledge_base("home loan documents required")
        → Returns 3 chunks about home loan documentation
    """
    kb = get_knowledge_base()

    # similarity_search_with_score returns (Document, score) tuples
    results = kb.similarity_search_with_score(query, k=top_k)

    formatted = []
    for doc, score in results:
        formatted.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
        })

    logger.info(
        f"KB search: '{query[:50]}...' → "
        f"{len(formatted)} results, best score: {formatted[0]['score']:.4f}"
    )

    return formatted