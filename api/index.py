"""
Vercel Serverless Entrypoint
AI Medical Report Analyzer

This file is the single entrypoint for Vercel's Python runtime.
It imports and re-exports the FastAPI `app` object from the main module.
"""

from app.main import app

# Vercel expects a module-level `app` variable (ASGI handler)
# No additional configuration needed — FastAPI is ASGI-native.
