from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChunkCitation(BaseModel):
    id: str
    text: str
    score: float
    strategy: str
    metadata: Dict[str, Any]

class LatencyTrace(BaseModel):
    stt: float = 0.0
    retrieval: float = 0.0
    guardrails_pre: float = 0.0
    generation: float = 0.0
    guardrails_post: float = 0.0
    total: float = 0.0

class GuardrailStatus(BaseModel):
    passed_pre: bool
    passed_post: bool
    is_safe: bool = True
    is_on_topic: bool = True
    is_grounded: bool = True
    refusal_reason: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: List[ChunkCitation] = []
    latency_trace: LatencyTrace
    guardrail_status: GuardrailStatus
    success: bool
    error_message: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    details: Dict[str, Any]

class MetricPoint(BaseModel):
    id: int
    timestamp: str
    query: str
    answer: str
    success: bool
    stt_latency: float
    retrieval_latency: float
    pre_guardrail_latency: float
    generation_latency: float
    post_guardrail_latency: float
    total_latency: float
    stt_provider: Optional[str] = None
    llm_provider: Optional[str] = None
    tokens_used: int = 0
    refused: bool = False

class BenchmarkSummary(BaseModel):
    p50: float
    p70: float
    p100: float
    total_requests: int
    success_rate: float
    avg_latency: float
    recent_runs: List[MetricPoint]
