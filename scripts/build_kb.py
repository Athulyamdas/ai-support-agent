"""
scripts/build_kb.py — One-time script to build the FAISS knowledge base index.

WHEN TO RUN THIS:
─────────────────
Run this script:
  1. On Day 3 setup (first time, to build the index)
  2. Any time you add or update files in data/knowledge_base/
  3. After changing CHUNK_SIZE or CHUNK_OVERLAP settings

You do NOT need to run this every time you start the Flask server.
The index is saved to disk and loaded automatically.

Usage:
    python scripts/build_kb.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.knowledge_base import build_knowledge_base
from app.utils.logger import get_logger

logger = get_logger("build_kb")


def main():
    print("\n" + "─" * 60)
    print("  Building Banking Knowledge Base")
    print("  This may take 30-60 seconds on first run")
    print("  (downloads embedding model ~80MB if not cached)")
    print("─" * 60 + "\n")

    try:
        vector_store = build_knowledge_base()

        # Quick test search to verify it works
        print("\n--- Running test search ---")
        from app.agent.knowledge_base import search_knowledge_base

        test_queries = [
            "home loan documents required",
            "credit card late payment charges",
            "how to report fraudulent transaction",
        ]

        for query in test_queries:
            results = search_knowledge_base(query, top_k=1)
            print(f"\nQuery: '{query}'")
            print(f"Best match (score={results[0]['score']:.4f}):")
            print(f"  {results[0]['content'][:150]}...")
            print(f"  Source: {results[0]['source']}")

        print("\n" + "─" * 60)
        print("  Knowledge base built and verified successfully!")
        print("  You can now start the Flask server: python main.py")
        print("─" * 60 + "\n")

    except Exception as e:
        logger.error(f"Failed to build knowledge base: {e}")
        raise


if __name__ == "__main__":
    main()