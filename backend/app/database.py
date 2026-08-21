import sqlite3
import os
import math
from datetime import datetime
from typing import Dict, Any, List
from backend.app.schemas import MetricPoint, BenchmarkSummary

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "metrics.db")

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            success INTEGER NOT NULL,
            stt_latency REAL NOT NULL,
            retrieval_latency REAL NOT NULL,
            pre_guardrail_latency REAL NOT NULL,
            generation_latency REAL NOT NULL,
            post_guardrail_latency REAL NOT NULL,
            total_latency REAL NOT NULL,
            stt_provider TEXT,
            llm_provider TEXT,
            tokens_used INTEGER DEFAULT 0,
            refused INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def log_request(
    query: str,
    answer: str,
    success: bool,
    stt_latency: float,
    retrieval_latency: float,
    pre_guardrail_latency: float,
    generation_latency: float,
    post_guardrail_latency: float,
    total_latency: float,
    stt_provider: str = None,
    llm_provider: str = None,
    tokens_used: int = 0,
    refused: bool = False
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO metrics (
            timestamp, query, answer, success,
            stt_latency, retrieval_latency, pre_guardrail_latency,
            generation_latency, post_guardrail_latency, total_latency,
            stt_provider, llm_provider, tokens_used, refused
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat() + "Z",
            query,
            answer,
            1 if success else 0,
            stt_latency,
            retrieval_latency,
            pre_guardrail_latency,
            generation_latency,
            post_guardrail_latency,
            total_latency,
            stt_provider,
            llm_provider,
            tokens_used,
            1 if refused else 0
        )
    )
    conn.commit()
    conn.close()

def get_percentile(sorted_list: List[float], percentile: float) -> float:
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_list[int(k)]
    d0 = sorted_list[int(f)] * (c - k)
    d1 = sorted_list[int(c)] * (k - f)
    return round(d0 + d1, 2)

def get_benchmark_summary(limit: int = 50) -> BenchmarkSummary:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get recent runs
    cursor.execute(
        "SELECT * FROM metrics ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    
    recent_runs = []
    for r in rows:
        recent_runs.append(
            MetricPoint(
                id=r["id"],
                timestamp=r["timestamp"],
                query=r["query"],
                answer=r["answer"],
                success=bool(r["success"]),
                stt_latency=r["stt_latency"],
                retrieval_latency=r["retrieval_latency"],
                pre_guardrail_latency=r["pre_guardrail_latency"],
                generation_latency=r["generation_latency"],
                post_guardrail_latency=r["post_guardrail_latency"],
                total_latency=r["total_latency"],
                stt_provider=r["stt_provider"],
                llm_provider=r["llm_provider"],
                tokens_used=r["tokens_used"],
                refused=bool(r["refused"])
            )
        )
        
    # Get all latency lists for percentiles (only where pipeline was successful)
    cursor.execute("SELECT total_latency FROM metrics WHERE success = 1")
    latencies = [row["total_latency"] for row in cursor.fetchall()]
    latencies.sort()
    
    # Counts
    cursor.execute("SELECT COUNT(*) as count FROM metrics")
    total_requests = cursor.fetchone()["count"]
    
    cursor.execute("SELECT COUNT(*) as count FROM metrics WHERE success = 1")
    successful_requests = cursor.fetchone()["count"]
    
    conn.close()
    
    success_rate = (successful_requests / total_requests) if total_requests > 0 else 1.0
    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    
    p50 = get_percentile(latencies, 0.50)
    p70 = get_percentile(latencies, 0.70)
    p100 = get_percentile(latencies, 1.0)
    
    return BenchmarkSummary(
        p50=p50,
        p70=p70,
        p100=p100,
        total_requests=total_requests,
        success_rate=round(success_rate * 100, 2),
        avg_latency=round(avg_latency, 2),
        recent_runs=recent_runs
    )

def clear_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM metrics")
    conn.commit()
    conn.close()
