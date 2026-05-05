"""
app/__init__.py — Flask application factory.

WHAT CHANGED FROM DAY 1?
─────────────────────────
Two additions:
  1. init_db() is called at startup — creates SQLite tables if missing
  2. admin_bp is registered — adds the /api/admin/* endpoints
"""

from flask import Flask
from flask_cors import CORS

import config


def create_app() -> Flask:
    config.validate()

    app = Flask(__name__)
    app.secret_key = config.FLASK_SECRET_KEY

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── NEW: Initialise the database (creates tables if they don't exist) ──
    from app.storage.database import init_db
    init_db()

    # Register blueprints (groups of related routes)
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    # ── NEW: Register admin blueprint for history inspection ──
    from app.api.admin_routes import admin_bp
    app.register_blueprint(admin_bp)

    return app