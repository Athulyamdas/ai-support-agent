"""
app/api/crm_routes.py — REST endpoints for CRM customer lookups.

WHY THESE ENDPOINTS?
─────────────────────
These endpoints let you test CRM lookups directly from Postman
without going through the chat interface. Useful for:
  - Verifying a customer exists before demoing
  - Admin lookups during portfolio demo
  - Testing the CRM service independently

ENDPOINTS:
  GET /api/crm/customers                    — list all customers (summary)
  GET /api/crm/customers/email/<email>      — lookup by email
  GET /api/crm/customers/phone/<phone>      — lookup by phone
  GET /api/crm/customers/<customer_id>      — lookup by customer ID
"""

from flask import Blueprint, jsonify, Response

from app.api.crm_service import (
    find_customer_by_email,
    find_customer_by_phone,
    format_customer_context,
    _load_crm,
    _customers,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
crm_bp = Blueprint("crm", __name__, url_prefix="/api/crm")


def _error(message: str, status: int = 400) -> tuple[Response, int]:
    return jsonify({"error": message}), status


@crm_bp.route("/customers", methods=["GET"])
def list_customers() -> tuple[Response, int]:
    """
    List all customers with summary info only.
    Used to see available test customers during demo.
    """
    _load_crm()
    summary = []
    for c in _customers:
        summary.append({
            "customer_id": c["customer_id"],
            "name": c["name"],
            "email": c["email"],
            "phone": c["phone"],
            "account_status": c["account_status"],
            "kyc_status": c["kyc_status"],
            "open_tickets": c["open_tickets"],
        })
    return jsonify({"customers": summary, "total": len(summary)}), 200


@crm_bp.route("/customers/email/<path:email>", methods=["GET"])
def get_by_email(email: str) -> tuple[Response, int]:
    """Look up a customer by email address."""
    customer = find_customer_by_email(email)
    if not customer:
        return _error(f"No customer found with email: {email}", 404)
    return jsonify({
        "customer": customer,
        "formatted_context": format_customer_context(customer),
    }), 200


@crm_bp.route("/customers/phone/<phone>", methods=["GET"])
def get_by_phone(phone: str) -> tuple[Response, int]:
    """Look up a customer by phone number."""
    customer = find_customer_by_phone(phone)
    if not customer:
        return _error(f"No customer found with phone: {phone}", 404)
    return jsonify({
        "customer": customer,
        "formatted_context": format_customer_context(customer),
    }), 200


@crm_bp.route("/customers/<customer_id>", methods=["GET"])
def get_by_id(customer_id: str) -> tuple[Response, int]:
    """Look up a customer by customer ID."""
    _load_crm()
    customer = next(
        (c for c in _customers if c["customer_id"] == customer_id.upper()),
        None
    )
    if not customer:
        return _error(f"No customer found with ID: {customer_id}", 404)
    return jsonify({
        "customer": customer,
        "formatted_context": format_customer_context(customer),
    }), 200