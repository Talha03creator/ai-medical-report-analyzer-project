"""
Centralized Configuration Module
AI Medical Report Analyzer

Loads all settings from environment variables with validation.
Never hardcodes sensitive values - all secrets via .env
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, computed_field
from functools import lru_cache
from typing import List, Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Pydantic validates all fields at startup - fails fast on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────
    app_name: str = "AI Medical Report Analyzer"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = False
    secret_key: str = "change-me-in-production-use-random-32-char-string"

    # ── Server ─────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Database ───────────────────────────────────────────────────
    # Default: in-memory SQLite (serverless-safe). Override with DATABASE_URL env var
    # for hosted PostgreSQL (e.g., Supabase, Neon).
    database_url: str = "sqlite+aiosqlite:///:memory:"
    database_sync_url: str = "sqlite:///:memory:"

    # ── AI Configuration ───────────────────────────────────────────
    medical_ai_api_key: str = ""
    ai_model: str = "gpt-4o-mini"
    ai_temperature: float = 0.2
    ai_max_tokens: int = 4096
    ai_base_url: Optional[str] = None
    ai_chunk_size: int = 3000          # tokens per chunk
    ai_max_retries: int = 3
    ai_retry_base_delay: float = 1.0   # seconds

    # ── Redis Cache ────────────────────────────────────────────────
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # ── Rate Limiting ──────────────────────────────────────────────
    rate_limit_requests: int = 5
    rate_limit_window: int = 60  # seconds

    # ── File Upload ────────────────────────────────────────────────
    max_file_size_mb: int = 10
    allowed_extensions: str = "txt,pdf"

    # ── CORS ───────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000,https://*.vercel.app"

    # ── Logging ────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"

    # ── System Disclaimer (immutable) ─────────────────────────────
    disclaimer: str = (
        "This system is for informational purposes only and does not "
        "provide medical diagnosis."
    )

    # ── Computed Properties ────────────────────────────────────────
    @computed_field
    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @computed_field
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @computed_field
    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    # ── Validators ─────────────────────────────────────────────────
    @field_validator("medical_ai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Warn but don't fail at import time — fail at runtime if missing."""
        return v

    @field_validator("ai_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("AI temperature must be between 0.0 and 1.0")
        return v

    def get_ai_api_key(self) -> str:
        """Get API key with explicit error if not configured."""
        key = self.medical_ai_api_key or os.environ.get("MEDICAL_AI_API_KEY", "")
        if not key:
            raise ValueError(
                "MEDICAL_AI_API_KEY environment variable is not set. "
                "Please configure your AI provider API key in the .env file."
            )
        return key

    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings instance — singleton pattern.
    Called once at startup, cached for lifetime of application.
    """
    return Settings()


# Module-level convenience access
settings = get_settings()
