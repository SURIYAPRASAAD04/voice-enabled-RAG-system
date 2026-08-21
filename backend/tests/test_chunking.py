import pytest
from backend.app.chunking import FixedSizeChunker, SemanticChunker, MetadataAwareChunker

FIXTURE_TEXT = (
    "This is the first sentence of our mock documentation. It is intended to test chunking splits. "
    "We want to verify that the chunks are generated correctly. Here is another sentence. "
    "Additionally, we are checking Hindi markers. यह टास्क टू का भाग है। यह स्वर-सक्षम आरएजी मॉडल है। "
    "We need to ensure boundaries and overlaps operate according to the layout configurations."
)

def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(FIXTURE_TEXT, "test_doc", {"category": "test"})
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["strategy"] == "fixed"
        assert chunk["metadata"]["doc_id"] == "test_doc"
        assert chunk["metadata"]["category"] == "test"
        assert "chunk_index" in chunk["metadata"]
        assert len(chunk["text"]) <= 100

def test_semantic_chunker_fallback():
    # If SentenceTransformers is not loaded, semantic chunker degrades to sentence length grouping
    chunker = SemanticChunker(max_chunk_size=150)
    chunks = chunker.chunk(FIXTURE_TEXT, "test_doc", {"category": "test"})
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["strategy"] == "semantic"
        assert chunk["metadata"]["doc_id"] == "test_doc"

def test_metadata_aware_chunker():
    base_chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    chunker = MetadataAwareChunker(base_chunker=base_chunker)
    metadata = {"language": "hi", "source": "msmarco", "title": "Test Title"}
    chunks = chunker.chunk(FIXTURE_TEXT, "test_doc", metadata)
    
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["strategy"] == "metadata"
        assert chunk["text"].startswith("[Doc: Test Title | Lang: hi | Source: msmarco]")
        assert chunk["metadata"]["injected_metadata"] is True
        assert "original_text" in chunk["metadata"]
