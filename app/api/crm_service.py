"""
app/agent/crm_service.py — Customer lookup from the mock CRM JSON file.

WHAT IS THIS FILE?
──────────────────
This is the CRM (Customer Relationship Management) layer.
In a real bank, this would query an Oracle or SAP CRM database.
For our project, it reads from data/mock_crm/customers.json.

WHAT DOES IT DO?
────────────────
1. Loads all customer records into memory on first use
2. Lets you look up a customer by email or phone number
3. Formats customer data into a readable summary for the LLM
4. Detects when a user message contains an email or phone number

WHY DOES THE LLM NEED FORMATTED TEXT?
───────────────────────────────────────
The LLM cannot read a Python dict directly in a meaningful way.
We convert the dict into a clean text summary that the LLM can
read and use to compose a personalised, accurate response.

Example input dict:
  { "name": "Alice", "balance": 45230.50, "account_status": "active" }

Example formatted output for LLM:
  Customer Name: Alice Johnson
  Account Status: ACTIVE
  Current Balance: Rs. 45,230.50
  ...
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── In-memory CRM store ───────────────────────────────────────────────────────
# Loaded once from JSON, kept in RAM for fast lookups
_customers: list[dict] = []
_crm_loaded: bool = False


def _load_crm() -> None:
    """Load the mock CRM JSON file into memory."""
    global _customers, _crm_loaded

    if _crm_loaded:
        return

    crm_path = Path(config.MOCK_CRM_PATH)

    if not crm_path.exists():
        logger.warning(f"CRM file not found at {crm_path}. CRM lookups will be unavailable.")
        _crm_loaded = True
        return

    with open(crm_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _customers = data.get("customers", [])
    _crm_loaded = True
    logger.info(f"CRM loaded: {len(_customers)} customers")


# ── Lookup functions ──────────────────────────────────────────────────────────

def find_customer_by_email(email: str) -> Optional[dict]:
    """
    Find a customer by their email address (case-insensitive).

    SQL equivalent: SELECT * FROM customers WHERE LOWER(email) = LOWER(?)
    """
    _load_crm()
    email = email.lower().strip()
    for customer in _customers:
        if customer.get("email", "").lower() == email:
            logger.info(f"CRM match by email: {email} → {customer['name']}")
            return customer
    logger.info(f"CRM: No customer found for email: {email}")
    return None


def find_customer_by_phone(phone: str) -> Optional[dict]:
    """
    Find a customer by their phone number.
    Strips spaces and dashes before comparing.
    """
    _load_crm()
    phone = re.sub(r"[\s\-]", "", phone)
    for customer in _customers:
        stored = re.sub(r"[\s\-]", "", customer.get("phone", ""))
        if stored == phone:
            logger.info(f"CRM match by phone: {phone} → {customer['name']}")
            return customer
    logger.info(f"CRM: No customer found for phone: {phone}")
    return None


def find_customer_by_account_number(account_number: str) -> Optional[dict]:
    """Find a customer by their account number."""
    _load_crm()
    account_number = account_number.strip().upper()
    for customer in _customers:
        if customer.get("account_number", "").upper() == account_number:
            logger.info(f"CRM match by account: {account_number} → {customer['name']}")
            return customer
    return None


# ── Identifier detection ──────────────────────────────────────────────────────

def extract_identifier_from_message(message: str) -> tuple[str, str] | tuple[None, None]:
    """
    Scan a user message for a customer identifier.

    Looks for:
      - Email address  (e.g. alice@example.com)
      - 10-digit phone (e.g. 9876543210)
      - Account number (e.g. SB-001-2023)

    Returns:
      (identifier_type, identifier_value) if found
      (None, None) if nothing found

    Examples:
      "my email is alice@example.com" → ("email", "alice@example.com")
      "my number is 9876543210"       → ("phone", "9876543210")
      "account SB-001-2023"           → ("account", "SB-001-2023")
    """
    # Email pattern
    email_match = re.search(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        message
    )
    if email_match:
        return "email", email_match.group()

    # 10-digit phone number
    phone_match = re.search(r'\b[6-9]\d{9}\b', message)
    if phone_match:
        return "phone", phone_match.group()

    # Account number pattern (SB-XXX-YYYY)
    account_match = re.search(r'\bSB-\d{3}-\d{4}\b', message, re.IGNORECASE)
    if account_match:
        return "account", account_match.group()

    return None, None


def lookup_customer_from_message(message: str) -> Optional[dict]:
    """
    Try to identify and fetch a customer from a user message.

    This is the main function called by the conversation chain.
    It combines extraction + lookup in one call.

    Returns the customer dict if found, None otherwise.
    """
    id_type, id_value = extract_identifier_from_message(message)

    if id_type is None:
        return None

    if id_type == "email":
        return find_customer_by_email(id_value)
    elif id_type == "phone":
        return find_customer_by_phone(id_value)
    elif id_type == "account":
        return find_customer_by_account_number(id_value)

    return None


# ── Context formatter ─────────────────────────────────────────────────────────

def format_customer_context(customer: dict) -> str:
    """
    Convert a customer dict into a readable text summary for the LLM.

    This text gets injected into the system prompt so Aria can
    answer account-specific questions accurately.

    IMPORTANT: We never expose full account numbers, full card numbers,
    or date of birth in the LLM context — only the last 4 digits of cards
    and summary-level information. This is a security best practice.
    """
    lines = [
        "═══ CUSTOMER ACCOUNT INFORMATION ═══",
        f"Customer Name    : {customer['name']}",
        f"Customer ID      : {customer['customer_id']}",
        f"Account Number   : {customer['account_number']}",
        f"Account Type     : {customer['account_type']}",
        f"Account Status   : {customer['account_status'].upper()}",
        f"Branch           : {customer.get('branch', 'N/A')}",
        f"KYC Status       : {customer['kyc_status'].upper()}",
        f"KYC Due Date     : {customer.get('kyc_due_date', 'N/A')}",
        f"Current Balance  : Rs. {customer['balance']:,.2f}",
        f"CIBIL Score      : {customer.get('cibil_score', 'N/A')}",
        f"Open Tickets     : {customer['open_tickets']}",
        f"Last Login       : {customer['last_login']}",
    ]

    # Cards section
    cards = customer.get("cards", [])
    if cards:
        lines.append("\nCARDS:")
        for card in cards:
            card_line = (
                f"  - {card['card_type']} Card "
                f"(****{card['card_number_last4']}) "
                f"Status: {card['status'].upper()}"
            )
            if card['card_type'] == 'Credit':
                card_line += (
                    f" | Limit: Rs. {card['credit_limit']:,} "
                    f"| Outstanding: Rs. {card['outstanding']:,}"
                )
                if card.get('due_date'):
                    card_line += f" | Due: {card['due_date']}"
                    card_line += f" | Min Due: Rs. {card['min_due']:,}"
            lines.append(card_line)

    # Loans section
    loans = customer.get("loans", [])
    if loans:
        lines.append("\nLOANS:")
        for loan in loans:
            loan_line = (
                f"  - {loan['loan_type']} ({loan['loan_id']}) "
                f"| Outstanding: Rs. {loan['outstanding']:,} "
                f"| EMI: Rs. {loan['emi']:,} on {loan['emi_due_date']} "
                f"| Status: {loan['status'].upper()}"
            )
            if loan.get('overdue_amount', 0) > 0:
                loan_line += (
                    f" ⚠ OVERDUE: Rs. {loan['overdue_amount']:,} "
                    f"({loan['overdue_months']} months)"
                )
            lines.append(loan_line)
    else:
        lines.append("\nLOANS: No active loans")

    lines.append("═══════════════════════════════════")

    return "\n".join(lines)