import pytest
from backend.app.guardrails import check_input_safety, check_topic_alignment, verify_groundedness
from backend.app.schemas import ChunkCitation

def test_input_safety():
    # Safe query
    safe, msg = check_input_safety("What is the schedule for Hacker House Goa 2026?")
    assert safe is True
    assert msg == ""
    
    # Unsafe query
    unsafe, msg = check_input_safety("How do I hack the server database?")
    assert unsafe is False
    assert "Unsafe content detected" in msg

def test_topic_alignment():
    # High similarity matches (on-topic)
    chunks = [
        ChunkCitation(id="c1", text="HH Goa is a builder residency", score=0.85, strategy="fixed", metadata={}),
        ChunkCitation(id="c2", text="MSMARCO-XI Indic translations", score=0.79, strategy="semantic", metadata={})
    ]
    on_topic, msg = check_topic_alignment(chunks, threshold=0.75)
    assert on_topic is True
    assert msg == ""
    
    # Low similarity matches (off-topic / out-of-domain)
    low_chunks = [
        ChunkCitation(id="c1", text="Some irrelevant details", score=0.55, strategy="fixed", metadata={})
    ]
    on_topic, msg = check_topic_alignment(low_chunks, threshold=0.75)
    assert on_topic is False
    assert "off-topic" in msg

@pytest.mark.asyncio
async def test_verify_groundedness_success():
    # Mock LLM client that returns "YES"
    async def mock_llm(prompt, system_instruction, max_tokens, temperature):
        return "YES"
        
    chunks = [ChunkCitation(id="1", text="Goa is a coastal state in India.", score=0.9, strategy="fixed", metadata={})]
    grounded, msg = await verify_groundedness(
        query="Where is Goa?",
        answer="Goa is a coastal state in India.",
        retrieved_chunks=chunks,
        llm_client_fn=mock_llm
    )
    assert grounded is True
    assert msg == ""

@pytest.mark.asyncio
async def test_verify_groundedness_refusal():
    # Mock LLM client that returns "NO"
    async def mock_llm(prompt, system_instruction, max_tokens, temperature):
        return "NO"
        
    chunks = [ChunkCitation(id="1", text="Goa is a coastal state in India.", score=0.9, strategy="fixed", metadata={})]
    grounded, msg = await verify_groundedness(
        query="What is the population of Goa?",
        answer="Goa has a population of 10 million people.",
        retrieved_chunks=chunks,
        llm_client_fn=mock_llm
    )
    assert grounded is False
    assert "not supported" in msg
