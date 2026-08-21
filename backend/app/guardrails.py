import logging
from typing import List, Tuple
from backend.app.schemas import ChunkCitation
from backend.app.config import settings

logger = logging.getLogger(__name__)

# Basic safety blocklist for demonstration
SAFETY_BLOCKLIST = [
    "hack the server", "sql injection", "drop database", "sudo rm",
    "kill myself", "suicide", "bomb", "terrorist", "make a weapon",
    "fuck", "shit", "asshole", "bastard"
]

def check_input_safety(query: str) -> Tuple[bool, str]:
    """Check query text against simple safety rules."""
    query_lower = query.lower()
    for block in SAFETY_BLOCKLIST:
        if block in query_lower:
            logger.warning(f"Safety check triggered for term: '{block}'")
            return False, f"Unsafe content detected (blocked keyword: '{block}')"
    return True, ""

def check_topic_alignment(retrieved_chunks: List[ChunkCitation], threshold: float = 0.72) -> Tuple[bool, str]:
    """Check if the query matches the indexed dataset semantically."""
    if not retrieved_chunks:
        return False, "Query retrieved zero relevant context."
        
    # Multilingual E5-small cosine scores are generally >= 0.70 for matches.
    # We inspect the maximum score of the retrieved chunks.
    max_score = max(chunk.score for chunk in retrieved_chunks)
    logger.info(f"Max retrieval similarity score: {max_score}")
    
    if max_score < threshold:
        return False, f"Query appears to be off-topic. (Max relevance score {max_score} < threshold {threshold})"
        
    return True, ""

async def verify_groundedness(
    query: str, 
    answer: str, 
    retrieved_chunks: List[ChunkCitation],
    llm_client_fn
) -> Tuple[bool, str]:
    """Runs a fast LLM-as-a-judge check to verify that the answer is supported by the context."""
    if not retrieved_chunks:
        return False, "No context available to ground the answer."
        
    # Format chunks context
    context_str = "\n\n".join(f"[Chunk {i}]: {c.text}" for i, c in enumerate(retrieved_chunks))
    
    # We use a strict prompt to check entailment
    prompt = f"""
[Context]
{context_str}

[User Query]
{query}

[Generated Answer]
{answer}

Task: Verify if the 'Generated Answer' is fully supported and grounded by the provided 'Context'.
Evaluate if there are any hallucinated claims or information not present in the Context.
You must reply in exactly one of the following formats:
- YES: The answer is fully supported by the context.
- PARTIAL: The answer is partially supported, but contains claims not found in the context.
- NO: The answer is not supported, or contradicts the context.

Reply with only one word (YES, PARTIAL, or NO). Do not include any explanations.
"""
    try:
        # We call the LLM function with low max_tokens to keep latency ultra-low (< 100ms on Groq)
        judge_response = await llm_client_fn(
            prompt=prompt,
            system_instruction="You are an unbiased, strict fact-checking assistant.",
            max_tokens=5,
            temperature=0.0
        )
        
        judge_decision = judge_response.strip().upper()
        logger.info(f"Groundedness judge decision: '{judge_decision}'")
        
        if "YES" in judge_decision:
            return True, ""
        elif "PARTIAL" in judge_decision:
            return False, "The answer contains ungrounded claims not supported by the context."
        else:
            return False, "The answer is not supported by the retrieved context."
            
    except Exception as e:
        logger.error(f"Groundedness verification LLM call failed: {e}")
        # In case of LLM-as-a-judge API failure, we gracefully degrade or reject based on policy.
        # Here we choose to fail open if the generated text contains citations, or fail closed.
        # Let's fail closed for high reliability during judge demos (avoids hallucinations).
        return False, f"Groundedness check failed to execute: {e}"
