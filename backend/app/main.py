import os
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis

from backend.app.config import settings
from backend.app.schemas import QueryResponse, HealthStatus
from backend.app.database import init_db, get_benchmark_summary, clear_db
from backend.app.retrieval import init_qdrant_collection, get_qdrant_client
from backend.app.rag import run_rag_pipeline
import httpx

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("backend_main")

app = FastAPI(
    title="HH Goa 2026 Task #2 Voice-RAG API",
    description="Backend API for Voice-Enabled RAG Pipeline",
    version="1.0.0"
)

# CORS Setup - restrict to allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiter & Daily Cap class
class RateLimiter:
    def __init__(self, host: str, port: int):
        self.use_redis = False
        try:
            self.redis_client = redis.Redis(host=host, port=port, socket_timeout=0.5)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("Connected to Redis successfully for rate limiting and daily cap.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Falling back to in-memory tracking.")
            self.in_memory_limits = {}  # IP -> list of timestamps
            self.daily_count = 0
            self.last_date = datetime.utcnow().date()

    def check_rate_limit(self, client_ip: str) -> bool:
        now = time.time()
        if self.use_redis:
            try:
                key = f"rate_limit:{client_ip}"
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, 60)
                count, _ = pipe.execute()
                return count <= settings.RATE_LIMIT_PER_MIN
            except Exception as e:
                logger.error(f"Redis rate limit check error: {e}")
                return self._check_in_memory_limit(client_ip, now)
        else:
            return self._check_in_memory_limit(client_ip, now)

    def _check_in_memory_limit(self, client_ip: str, now: float) -> bool:
        if client_ip not in self.in_memory_limits:
            self.in_memory_limits[client_ip] = []
        self.in_memory_limits[client_ip] = [t for t in self.in_memory_limits[client_ip] if now - t < 60]
        if len(self.in_memory_limits[client_ip]) >= settings.RATE_LIMIT_PER_MIN:
            return False
        self.in_memory_limits[client_ip].append(now)
        return True

    def check_daily_cap(self) -> bool:
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        if self.use_redis:
            try:
                key = f"daily_cap:{today_str}"
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, 172800)  # 2 days expiry
                count, _ = pipe.execute()
                return count <= settings.DAILY_REQUEST_CAP
            except Exception as e:
                logger.error(f"Redis daily cap check error: {e}")
                return self._check_in_memory_daily_cap()
        else:
            return self._check_in_memory_daily_cap()

    def _check_in_memory_daily_cap(self) -> bool:
        today = datetime.utcnow().date()
        if today != self.last_date:
            self.daily_count = 0
            self.last_date = today
        self.daily_count += 1
        return self.daily_count <= settings.DAILY_REQUEST_CAP

limiter = RateLimiter(settings.REDIS_HOST, settings.REDIS_PORT)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing metrics SQLite database...")
    init_db()
    
    logger.info("Initializing Qdrant connection and collection...")
    init_qdrant_collection()

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text_query: Optional[str] = Form(None),
    browser_transcript: Optional[str] = Form(None)
):
    # Retrieve client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # 1. Rate Limiting Check
    if not limiter.check_rate_limit(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 10 requests per minute."
        )
        
    # 2. Daily Cost Cap Check
    if not limiter.check_daily_cap():
        logger.warning("Daily request cap exceeded.")
        # Return a friendly refusal instead of an error crash
        return QueryResponse(
            query="",
            answer="Demo daily limit reached for today. Cost guardrails have paused paid API calls.",
            retrieved_chunks=[],
            latency_trace=LatencyTrace(),
            guardrail_status=GuardrailStatus(passed_pre=False, passed_post=False, refusal_reason="Daily request limit reached"),
            success=False,
            error_message="Daily cost cap exceeded."
        )

    # 3. Audio or Text ingestion
    audio_bytes = None
    filename = "query.wav"
    content_type = "audio/wav"
    
    if file is not None:
        # Check size constraints
        audio_content = await file.read()
        if len(audio_content) > 10 * 1024 * 1024:  # 10MB Limit
            raise HTTPException(status_code=400, detail="Audio file too large. Max size is 10MB.")
        audio_bytes = audio_content
        filename = file.filename or "query.wav"
        content_type = file.content_type or "audio/wav"
        
    response = await run_rag_pipeline(
        audio_bytes=audio_bytes,
        text_query=text_query,
        browser_transcript=browser_transcript,
        filename=filename,
        content_type=content_type
    )
    
    return response

@app.get("/api/health", response_model=HealthStatus)
async def health_endpoint():
    status_str = "healthy"
    details = {}
    
    # 1. Qdrant reachable
    try:
        q_client = get_qdrant_client()
        q_client.get_collections()
        details["qdrant"] = "connected"
    except Exception as e:
        status_str = "unhealthy"
        details["qdrant"] = f"error: {str(e)}"
        
    # 2. Redis reachable
    try:
        r_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, socket_timeout=0.2)
        r_client.ping()
        details["redis"] = "connected"
    except Exception as e:
        # We don't mark the whole app unhealthy if only Redis is down (it degrades to in-memory)
        details["redis"] = f"error: {str(e)} (degraded to in-memory)"
        
    # 3. STT Provider Config & reachability
    stt_provider = settings.STT_PROVIDER.lower()
    details["stt_provider"] = stt_provider
    if stt_provider == "sarvam":
        if settings.SARVAM_API_KEY:
            details["stt_auth"] = "key_configured"
            # Lightweight reachability check
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    res = await client.get("https://api.sarvam.ai/")
                    # Even if 404 or 401, it is reachable
                    details["stt_endpoint"] = "reachable" if res.status_code < 500 else f"status_code_{res.status_code}"
            except Exception as e:
                details["stt_endpoint"] = f"unreachable: {str(e)}"
        else:
            details["stt_auth"] = "key_missing"
    elif stt_provider == "elevenlabs":
        if settings.ELEVENLABS_API_KEY:
            details["stt_auth"] = "key_configured"
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    res = await client.get("https://api.elevenlabs.io/v1/user", headers={"xi-api-key": settings.ELEVENLABS_API_KEY})
                    details["stt_endpoint"] = "reachable_and_authorized" if res.status_code == 200 else f"unauthorized_status_{res.status_code}"
            except Exception as e:
                details["stt_endpoint"] = f"unreachable: {str(e)}"
        else:
            details["stt_auth"] = "key_missing"
            
    # 4. LLM Provider Config & reachability
    llm_provider = settings.LLM_PROVIDER.lower()
    details["llm_provider"] = llm_provider
    if llm_provider == "openai":
        if settings.OPENAI_API_KEY:
            details["llm_auth"] = "key_configured"
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    res = await client.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"})
                    details["llm_endpoint"] = "reachable_and_authorized" if res.status_code == 200 else f"status_{res.status_code}"
            except Exception as e:
                details["llm_endpoint"] = f"unreachable: {str(e)}"
        else:
            details["llm_auth"] = "key_missing"
    elif llm_provider == "groq":
        if settings.GROQ_API_KEY:
            details["llm_auth"] = "key_configured"
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    res = await client.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"})
                    details["llm_endpoint"] = "reachable_and_authorized" if res.status_code == 200 else f"status_{res.status_code}"
            except Exception as e:
                details["llm_endpoint"] = f"unreachable: {str(e)}"
        else:
            details["llm_auth"] = "key_missing"
    elif llm_provider == "anthropic":
        if settings.ANTHROPIC_API_KEY:
            details["llm_auth"] = "key_configured"
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    res = await client.get("https://api.anthropic.com/v1/messages", headers={"x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"})
                    # Will be 400 since no payload, but reachable
                    details["llm_endpoint"] = "reachable" if res.status_code < 500 else f"status_{res.status_code}"
            except Exception as e:
                details["llm_endpoint"] = f"unreachable: {str(e)}"
        else:
            details["llm_auth"] = "key_missing"
            
    if status_str == "unhealthy":
        raise HTTPException(status_code=500, detail={"status": "unhealthy", "details": details})
        
    return HealthStatus(status=status_str, details=details)

@app.get("/api/metrics")
async def metrics_endpoint():
    try:
        summary = get_benchmark_summary(limit=50)
        return summary
    except Exception as e:
        logger.error(f"Failed to fetch metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics/clear")
async def clear_metrics_endpoint():
    try:
        clear_db()
        return {"status": "success", "message": "Metrics database cleared successfully."}
    except Exception as e:
        logger.error(f"Failed to clear metrics database: {e}")
        raise HTTPException(status_code=500, detail=str(e))
