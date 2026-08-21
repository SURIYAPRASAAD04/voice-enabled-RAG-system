import pytest
from unittest.mock import patch, AsyncMock
from backend.app.rag import run_rag_pipeline
from backend.app.schemas import QueryResponse, ChunkCitation
from backend.app.database import init_db

@pytest.mark.asyncio
async def test_full_pipeline_with_mocks():
    # Initialize the database so that logging works during testing
    init_db()

    # Provide real ChunkCitation objects rather than mocks to pass Pydantic validation
    mock_chunks = [
        ChunkCitation(id="c1", text="Hacker House Goa 2026 takes place in October.", score=0.88, strategy="fixed", metadata={}),
        ChunkCitation(id="c2", text="Builders work in teams of 1-3 people.", score=0.82, strategy="semantic", metadata={})
    ]
    
    # Setup mock STT provider instance
    mock_stt_provider = AsyncMock()
    mock_stt_provider.transcribe.return_value = "When does Hacker House Goa take place?"
    
    # Mock hybrid retrieval, get_stt_provider, LLM call, and groundedness check
    with patch("backend.app.rag.hybrid_search", new_callable=AsyncMock) as mock_search, \
         patch("backend.app.rag.get_stt_provider") as mock_get_stt, \
         patch("backend.app.rag.call_llm", new_callable=AsyncMock) as mock_llm, \
         patch("backend.app.rag.verify_groundedness", new_callable=AsyncMock) as mock_verify:
         
        mock_search.return_value = mock_chunks
        mock_get_stt.return_value = mock_stt_provider
        mock_llm.return_value = "Hacker House Goa 2026 takes place in October [Source 1]."
        mock_verify.return_value = (True, "")
        
        # Trigger the pipeline with dummy audio
        response: QueryResponse = await run_rag_pipeline(
            audio_bytes=b"dummy_wav_content",
            filename="query.wav",
            content_type="audio/wav"
        )
        
        # Assert structure matches expectations
        assert response.success is True
        assert response.query == "When does Hacker House Goa take place?"
        assert "October" in response.answer
        assert "[Source 1]" in response.answer
        assert len(response.retrieved_chunks) == 2
        
        # Verify latency tracers recorded stages
        assert response.latency_trace.stt > 0.0
        assert response.latency_trace.retrieval > 0.0
        assert response.latency_trace.generation > 0.0
        assert response.latency_trace.total > 0.0
        
        # Check guardrails status
        assert response.guardrail_status.passed_pre is True
        assert response.guardrail_status.passed_post is True
        assert response.guardrail_status.is_safe is True
        assert response.guardrail_status.is_on_topic is True
        assert response.guardrail_status.is_grounded is True
