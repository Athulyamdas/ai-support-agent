"""
main.py — Application entry point.

Run:
    python main.py

Or with Flask's dev server:
    flask --app main:app run --reload
"""

from app import create_app
import config

app = create_app()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=config.FLASK_PORT,
        debug=(config.FLASK_ENV == "development"),
    )
