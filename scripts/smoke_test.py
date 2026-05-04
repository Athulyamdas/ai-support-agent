"""
scripts/smoke_test.py — Interactive CLI to verify Day 1 works end-to-end.

Does NOT require the Flask server to be running; hits the chain directly.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --session my-test-session
"""

from __future__ import annotations
import argparse
import sys
import uuid

# Make sure project root is on the path
sys.path.insert(0, __file__.rsplit("/scripts", 1)[0])

from app.agent.conversation_chain import chat, clear_session
from app.utils.logger import get_logger

logger = get_logger("smoke_test")


def run_automated_test(session_id: str) -> None:
    """Send a scripted conversation and print results."""
    turns = [
        "Hi, I can't log into my account. It says my password is wrong.",
        "I already tried resetting it twice but it's still not working.",
        "My email is alice@example.com. Can you check if there's a problem?",
        "How do I cancel my subscription if I can't log in?",
    ]

    print(f"\n{'─'*60}")
    print(f"  Automated smoke test  |  session: {session_id}")
    print(f"{'─'*60}\n")

    for i, message in enumerate(turns, 1):
        print(f"[Turn {i}] Customer: {message}")
        result = chat(session_id=session_id, user_message=message)
        print(f"[Turn {i}] Aria     : {result['reply']}")
        print(f"           (history_len={result['history_len']})\n")

    clear_session(session_id)
    print("✓ Session cleared. Smoke test complete.\n")


def run_interactive(session_id: str) -> None:
    """REPL-style chat loop."""
    print(f"\n{'─'*60}")
    print("  AI Support Agent — interactive mode")
    print(f"  session: {session_id}")
    print("  Type 'quit' or Ctrl-C to exit, 'clear' to reset memory.")
    print(f"{'─'*60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower() == "clear":
            clear_session(session_id)
            print("Memory cleared. Starting fresh.\n")
            continue

        result = chat(session_id=session_id, user_message=user_input)
        print(f"Aria: {result['reply']}")
        print(f"      [turn {result['turn']}, {result['history_len']} msgs in memory]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test the support agent chain.")
    parser.add_argument("--session", default=str(uuid.uuid4()), help="Session ID to use")
    parser.add_argument("--auto", action="store_true", help="Run automated test (no input needed)")
    args = parser.parse_args()

    if args.auto:
        run_automated_test(args.session)
    else:
        run_interactive(args.session)
