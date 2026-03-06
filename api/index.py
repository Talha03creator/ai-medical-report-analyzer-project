"""
Vercel Serverless Entrypoint
AI Medical Report Analyzer

This file is the single entrypoint for Vercel's Python runtime.
It imports and re-exports the FastAPI `app` object from the main module.
"""

import os
import sys

import os
import sys
import traceback

# Add the project root to the sys.path so 'app' can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from app.main import app
except Exception as e:
    # If the app fails to import, create a dummy ASGI app that returns the error
    error_traceback = traceback.format_exc()
    
    async def app(scope, receive, send):
        assert scope['type'] == 'http'
        await send({
            'type': 'http.response.start',
            'status': 200,
            'headers': [
                [b'content-type', b'text/plain'],
            ],
        })
        await send({
            'type': 'http.response.body',
            'body': f"Vercel Import Error:\n{error_traceback}".encode('utf-8'),
        })

