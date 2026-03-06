"""
Database Configuration & Session Management
AI Medical Report Analyzer

Async SQLAlchemy setup - supports both PostgreSQL and SQLite.
Serverless-compatible: uses NullPool for non-SQLite to avoid stale connections.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool, StaticPool
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from typing import AsyncGenerator
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Detect DB type ─────────────────────────────────────────────────
_is_sqlite = settings.database_url.startswith("sqlite")

# ── Engine (serverless-safe) ───────────────────────────────────────
try:
    if _is_sqlite:
        engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,  # single connection for SQLite
        )
    else:
        # NullPool: no persistent connections — ideal for serverless
        engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            poolclass=NullPool,
            pool_pre_ping=True,
        )
except Exception as e:
    logger.warning(f"Database engine creation failed: {e}")
    # Fallback to in-memory SQLite so the app can at least start
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

# ── Session Factory ────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ── Base Model ─────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


# ── Dependency ─────────────────────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async database session.
    Automatically handles commit/rollback/close.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Health Check ───────────────────────────────────────────────────
async def check_database_connection() -> bool:
    """Verify database connectivity."""
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# ── Table Initialization ───────────────────────────────────────────
async def init_db() -> None:
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")
