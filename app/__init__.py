"""
app/__init__.py — Flask application factory.

Using the factory pattern (create_app) makes the app testable and avoids
circular imports — a common LangChain + Flask footgun.
"""

from flask import Flask
from flask_cors import CORS

import config


def create_app() -> Flask:
    config.validate()          # fail fast if API keys are missing

    app = Flask(__name__)
    app.secret_key = config.FLASK_SECRET_KEY

    # Allow any origin during development; tighten in production
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp)

    return app