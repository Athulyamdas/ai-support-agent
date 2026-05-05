"""
app/storage/database.py — Database connection and table creation.

WHAT IS THIS FILE?
──────────────────
Think of this file as the "database administrator".
It does two things:
  1. Creates a connection to the SQLite database file
  2. Creates the tables if they don't exist yet

WHAT IS SQLITE?
───────────────
SQLite is a database that lives in a single FILE on your hard disk.
Unlike MySQL (which needs a running server), SQLite needs nothing extra.
The entire database is just one file: data/chat_history.db

You already know MySQL tables — SQLite works exactly the same way,
just in a file instead of a server.

WHAT IS SQLALCHEMY?
────────────────────
SQLAlchemy is a Python library that lets you talk to databases using
Python code instead of raw SQL strings. It works with SQLite, MySQL,
PostgreSQL — the same Python code works on all of them.
"""

from sqlalchemy import (
    create_engine,   # creates the database connection
    Column,          # defines a column in a table
    String,          # text data type
    Text,            # long text data type
    Integer,         # number data type
    DateTime,        # date and time data type
    inspect          # lets us check if tables exist
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone

import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Base class ────────────────────────────────────────────────────────────────
# All database table classes will inherit from this Base.
# SQLAlchemy uses this to track which classes map to which tables.
Base = declarative_base()


# ── Table Definitions ─────────────────────────────────────────────────────────

class SessionModel(Base):
    """
    Represents the 'sessions' table in the database.

    Each row = one conversation session.

    In MySQL Workbench terms, this is like:
    CREATE TABLE sessions (
        session_id   VARCHAR(100) PRIMARY KEY,
        created_at   DATETIME,
        updated_at   DATETIME,
        message_count INTEGER,
        status       VARCHAR(20)
    );
    """
    __tablename__ = "sessions"

    session_id    = Column(String(100), primary_key=True)
    created_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at    = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))
    message_count = Column(Integer, default=0)
    status        = Column(String(20), default="active")


class MessageModel(Base):
    """
    Represents the 'messages' table in the database.

    Each row = one message (either from the human or from the AI).

    In MySQL Workbench terms:
    CREATE TABLE messages (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id   VARCHAR(100),
        role         VARCHAR(20),    -- 'human' or 'ai'
        content      TEXT,
        created_at   DATETIME
    );
    """
    __tablename__ = "messages"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False)   # which conversation
    role       = Column(String(20),  nullable=False)   # 'human' or 'ai'
    content    = Column(Text,        nullable=False)   # the actual message text
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Engine and Session Factory ────────────────────────────────────────────────

# The engine is the connection to the database file.
# sqlite:///  means "SQLite file, relative path"
# config.SQLITE_DB_PATH is the path from your .env file
engine = create_engine(
    f"sqlite:///{config.SQLITE_DB_PATH}",
    connect_args={"check_same_thread": False},  # needed for Flask multi-threading
    echo=False,   # set True to see every SQL query in the terminal (useful for debugging)
)

# SessionFactory creates database sessions (not to be confused with chat sessions!).
# A database session = a unit of work with the database (like a transaction).
SessionFactory = sessionmaker(bind=engine)


def init_db() -> None:
    """
    Create all tables if they don't exist yet.

    This is safe to call every time the app starts —
    it checks first and only creates tables that are missing.

    In MySQL Workbench terms: this runs CREATE TABLE IF NOT EXISTS
    for every table defined above.
    """
    # Make sure the data/ folder exists
    config.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create all tables defined in Base's subclasses
    Base.metadata.create_all(engine)
    logger.info(f"Database initialised at: {config.SQLITE_DB_PATH}")


def get_db_session():
    """
    Get a database session to perform queries.

    Usage pattern (always use try/finally to close):
        db = get_db_session()
        try:
            db.query(...)
            db.commit()
        finally:
            db.close()
    """
    return SessionFactory()