"""
app/__init__.py — Flask application factory.

WHAT CHANGED FROM DAY 3?
─────────────────────────
Added: crm_bp registration for /api/crm/* endpoints
"""

from flask import Flask
from flask_cors import CORS
import config


def create_app() -> Flask:
    config.validate()

    app = Flask(__name__)
    app.secret_key = config.FLASK_SECRET_KEY

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialise database
    from app.storage.database import init_db
    init_db()

    # Register all blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    from app.api.admin_routes import admin_bp
    app.register_blueprint(admin_bp)

    # NEW Day 4: CRM endpoints
    from app.api.crm_routes import crm_bp
    app.register_blueprint(crm_bp)

    return app