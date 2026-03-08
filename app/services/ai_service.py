"""
AI Service — Google Gemini Integration (google-genai AsyncClient)
AI Medical Report Analyzer

Uses the native async client from google-genai SDK (no threading needed).
Model: gemini-2.5-flash (confirmed available for this API key)
"""

import re
import json
import time
import logging
import asyncio
import hashlib
from typing import Any, Optional

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """You are a medical document analyst. Extract structured information from the medical transcription below.

CRITICAL: Return ONLY a valid JSON object. No markdown, no code fences, no explanations.

Required JSON structure (keep very brief):
{{
  "patient_info": {{"age": "", "gender": ""}},
  "symptoms": [""],
  "medications": [""],
  "procedures": [""],
  "risk_flags": [""],
  "clinical_impression": "",
  "professional_summary": "",
  "confidence_score": 0.85
}}

Do NOT diagnose or prescribe. Use [] for empty lists, null for missing values.

MEDICAL TRANSCRIPTION:
{text}"""


# ── In-memory cache ───────────────────────────────────────────────
class _Cache:
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self._store: dict[str, tuple[Any, float]] = {}
        self._max = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry:
            val, ts = entry
            if time.time() - ts < self._ttl:
                return val
            del self._store[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self._max:
            oldest = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest]
        self._store[key] = (value, time.time())

    @staticmethod
    def make_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()


_cache = _Cache()


# ── JSON extraction ───────────────────────────────────────────────
def _extract_json(raw: str) -> Optional[dict]:
    """3-pass JSON extraction from Gemini response."""
    if not raw:
        return None
    # Pass 1: direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Pass 2: strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Pass 3: extract first {...} block
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning("JSON extraction failed. Preview: %.200s", raw)
    return None


# ── AI Service ────────────────────────────────────────────────────
class GeminiAIService:
    """Uses google-genai Client.aio for native async calls — no threading needed."""

    def __init__(self):
        self._client: Optional[genai.Client] = None
        self._model_name: str = "gemini-2.5-flash"

    def _ensure_initialized(self) -> None:
        """Create Client once. Raises ValueError if API key missing."""
        if self._client is not None:
            return
        api_key = settings.get_ai_api_key()
        self._client = genai.Client(api_key=api_key)
        self._model_name = getattr(settings, "ai_model", "gemini-2.5-flash")
        if self._model_name.startswith("models/"):
            self._model_name = self._model_name[len("models/"):]
        logger.info("Gemini client initialized — model: %s", self._model_name)

    # ── Single chunk call ────────────────────────────────────────
    async def _call_gemini(self, text: str) -> Optional[dict]:
        """Call Gemini async, retry once on failure."""
        prompt = ANALYSIS_PROMPT.format(text=text)
        config = types.GenerateContentConfig(
            temperature=float(getattr(settings, "ai_temperature", 0.2)),
            top_p=0.8,
            max_output_tokens=1024, # Reduced for speed
        )

        for attempt in range(1): # No retries on Vercel to avoid >10s timeout
            try:
                logger.info("Gemini call attempt %d", attempt + 1)
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self._model_name,
                        contents=prompt,
                        config=config,
                    ),
                    timeout=8.0, # Stop at 8s before Vercel kills us at 10s
                )
                raw = getattr(response, "text", "") or ""
                logger.info("Gemini response received (%d chars)", len(raw))

                parsed = _extract_json(raw)
                if parsed is not None:
                    return parsed

                logger.warning("Attempt %d: invalid JSON from Gemini", attempt + 1)

            except asyncio.TimeoutError:
                logger.error("Attempt %d: Gemini call timed out after 120s", attempt + 1)
            except Exception as exc:
                err = str(exc) or repr(exc)
                logger.error("Attempt %d: Gemini error — %s: %s", attempt + 1, type(exc).__name__, err)

            if attempt == 0:
                logger.info("Retrying in 2s...")
                await asyncio.sleep(2)

        return None

    # ── Chunking ─────────────────────────────────────────────────
    @staticmethod
    def _chunk_text(text: str, max_chars: int = 2500) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = s
            else:
                current = (current + " " + s).strip() if current else s
        if current:
            chunks.append(current.strip())
        logger.info("Split into %d chunk(s)", len(chunks))
        return chunks

    # ── Merge results ────────────────────────────────────────────
    @staticmethod
    def _merge(results: list[dict]) -> dict:
        if len(results) == 1:
            return results[0]

        def dedup(lst):
            seen, out = set(), []
            for x in lst:
                k = str(x).lower().strip()
                if k not in seen:
                    seen.add(k)
                    out.append(x)
            return out

        def best(vals):
            c = [v for v in vals if v and isinstance(v, str)]
            return max(c, key=len) if c else None

        return {
            "patient_info": {
                "age":    best([r.get("patient_info", {}).get("age")    for r in results]),
                "gender": best([r.get("patient_info", {}).get("gender") for r in results]),
            },
            "symptoms":                 dedup(sum([r.get("symptoms")    or [] for r in results], [])),
            "medications":              dedup(sum([r.get("medications") or [] for r in results], [])),
            "procedures":               dedup(sum([r.get("procedures")  or [] for r in results], [])),
            "lab_values":               dedup(sum([r.get("lab_values")  or [] for r in results], [])),
            "body_parts":               dedup(sum([r.get("body_parts")  or [] for r in results], [])),
            "risk_flags":               dedup(sum([r.get("risk_flags")  or [] for r in results], [])),
            "clinical_impression":      best([r.get("clinical_impression")      for r in results]),
            "specialty_classification": best([r.get("specialty_classification") for r in results]),
            "professional_summary":     best([r.get("professional_summary")     for r in results]),
            "patient_friendly_summary": best([r.get("patient_friendly_summary") for r in results]),
            "confidence_score": round(
                sum(r.get("confidence_score", 0.5) for r in results) / len(results), 2
            ),
        }

    # ── Public: analyze ──────────────────────────────────────────
    async def analyze_text(self, text: str) -> dict:
        self._ensure_initialized()

        # Cache check
        key = _cache.make_key(text)
        cached = _cache.get(key)
        if cached:
            logger.info("Cache HIT — returning stored result")
            return {**cached, "cached": True}

        chunks = self._chunk_text(text)
        results, failed = [], []

        for i, chunk in enumerate(chunks):
            logger.info("Processing chunk %d/%d (%d chars)...", i + 1, len(chunks), len(chunk))
            r = await self._call_gemini(chunk)
            if r:
                results.append(r)
                logger.info("Chunk %d/%d: OK", i + 1, len(chunks))
            else:
                failed.append(i + 1)
                logger.error("Chunk %d/%d: FAILED", i + 1, len(chunks))

        if not results:
            logger.error("All chunks failed. Count=%d", len(chunks))
            return {
                "status": "failed",
                "error": "AI analysis failed. Please try again later.",
                "patient_info": {}, "symptoms": [], "medications": [], "procedures": [],
                "lab_values": [], "body_parts": [], "clinical_impression": None,
                "risk_flags": [], "specialty_classification": None,
                "professional_summary": None, "patient_friendly_summary": None,
                "confidence_score": 0.0, "cached": False,
            }

        merged = self._merge(results)
        merged.update({"status": "success", "error": None, "cached": False})
        _cache.set(key, merged)
        logger.info("Analysis DONE — confidence=%.2f | chunks=%d/%d OK",
                    merged.get("confidence_score", 0), len(results), len(chunks))
        return merged

    # ── Public: test connection ───────────────────────────────────
    async def test_connection(self) -> dict:
        try:
            self._ensure_initialized()
            logger.info("Testing Gemini connection with model=%s", self._model_name)
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents="Say hello in one sentence.",
                ),
                timeout=30.0,
            )
            reply = getattr(response, "text", "no text") or "no text"
            logger.info("Gemini test OK: %s", reply[:80])
            return {"status": "ok", "model": self._model_name, "response": reply[:300]}
        except ValueError as e:
            return {"status": "error", "error": str(e)}
        except Exception as e:
            err = str(e) or repr(e)
            logger.error("LLM test failed — %s: %s", type(e).__name__, err)
            return {"status": "error", "error": err}

    async def close(self) -> None:
        self._client = None
        logger.info("AI service closed")


# Singleton
ai_service = GeminiAIService()
