
import logging
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles


# Force UTF-8 on Windows console to avoid emoji encoding crashes
# Guard: only on Windows AND only when stdout has a .buffer (not in serverless)
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # Serverless or CI environment — skip safely

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.database.session import init_db
from app.api.middleware.rate_limiter import RateLimitMiddleware
from app.api.middleware.logging_middleware import LoggingMiddleware
from app.api.routes import reports, health
from app.services.ai_service import ai_service


configure_logging()
logger = logging.getLogger(__name__)


# -- Lifespan (startup/shutdown) -----------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"  Environment: {settings.app_env}")
    logger.info(f"  AI Model:    {settings.ai_model}")

    # Validate AI API key at startup
    try:
        settings.get_ai_api_key()
        logger.info("  AI API key: configured")
    except ValueError as e:
        logger.warning(f"  AI API key WARNING: {e}")

    # Initialize database tables (auto-create in dev mode)
    try:
        await init_db()
        logger.info("  Database: initialized")
    except Exception as e:
        logger.warning(f"  Database WARNING: {e}")

    logger.info(f"  {settings.app_name} is ready at http://localhost:{settings.port}")
    yield

    # Cleanup
    logger.info("Shutting down...")
    await ai_service.close()
    logger.info("Shutdown complete.")


# -- FastAPI App ---------------------------------------------------------------
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered medical transcription analysis system. "
        "Extracts structured entities, classifies medical specialty, "
        "detects risk factors, and generates professional summaries.\n\n"
        "**DISCLAIMER:** This system is for informational purposes only "
        "and does not provide medical diagnosis."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# -- Middleware (outermost first) ----------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# -- Routes -------------------------------------------------------------------
app.include_router(health.router)
app.include_router(reports.router)


# -- Global Exception Handler -------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
            "disclaimer": settings.disclaimer,
        },
    )


# -- Frontend Static Files ----------------------------------------------------
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))


# -- Root API Info ------------------------------------------------------------
@app.get("/api", include_in_schema=False)
async def api_info():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "disclaimer": settings.disclaimer,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production(),
        log_level=settings.log_level.lower(),
    )
