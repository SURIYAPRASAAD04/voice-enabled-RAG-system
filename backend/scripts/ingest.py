import os
import sys
import logging
import uuid
from datasets import load_dataset
from qdrant_client.http import models

# Add current workspace directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.config import settings
from backend.app.retrieval import get_qdrant_client, init_qdrant_collection, COLLECTION_NAME
from backend.app.chunking import get_embedding_model, FixedSizeChunker, SemanticChunker, MetadataAwareChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")

def parse_passages(passages_field: Any) -> list:
    """Parse passages dynamically handling both list-of-dicts and dict-of-lists formats."""
    parsed = []
    if not passages_field:
        return parsed
        
    # Format 1: dict of lists: {"passage_text": ["txt1", "txt2"], "is_selected": [0, 1], ...}
    if isinstance(passages_field, dict):
        texts = passages_field.get("passage_text") or passages_field.get("text") or []
        urls = passages_field.get("url") or []
        for i, txt in enumerate(texts):
            url = urls[i] if i < len(urls) else "unknown"
            parsed.append({
                "text": txt,
                "url": url
            })
            
    # Format 2: list of dicts: [{"passage_text": "txt1", "url": "url1"}, ...]
    elif isinstance(passages_field, list):
        for item in passages_field:
            if isinstance(item, dict):
                txt = item.get("passage_text") or item.get("text")
                url = item.get("url") or "unknown"
                if txt:
                    parsed.append({
                        "text": txt,
                        "url": url
                    })
                    
    return parsed

def ingest_dataset(limit: int = 50, lang: str = "hi"):
    logger.info("Initializing database index structure...")
    if not init_qdrant_collection():
        logger.error("Could not initialize Qdrant collection. Exiting.")
        return
        
    client = get_qdrant_client()
    model = get_embedding_model()
    if model is None:
        logger.error("SentenceTransformers model failed to load. Exiting.")
        return
        
    logger.info(f"Loading HuggingFace dataset 'ai4bharat/MSMARCO-XI' ({lang} split) in streaming mode...")
    try:
        # Load in streaming mode to run fast and consume minimal resources
        dataset = load_dataset("ai4bharat/MSMARCO-XI", lang, split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Initialize Chunkers
    fixed_chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=100)
    semantic_chunker = SemanticChunker(similarity_threshold=0.65, max_chunk_size=800)
    meta_chunker = MetadataAwareChunker(base_chunker=fixed_chunker)

    logger.info("Starting ingestion processing...")
    records_processed = 0
    points_to_upload = []
    
    # Iterate and extract passages
    for item in dataset:
        if records_processed >= limit:
            break
            
        query_text = item.get("query", "")
        passages_field = item.get("passages")
        
        parsed_passages = parse_passages(passages_field)
        if not parsed_passages:
            continue
            
        doc_id = f"msmarco_{records_processed}"
        
        for p_idx, passage in enumerate(parsed_passages):
            passage_text = passage["text"]
            source_url = passage["url"]
            
            doc_metadata = {
                "language": lang,
                "source": "msmarco-xi",
                "source_url": source_url,
                "original_query": query_text,
                "title": f"MSMARCO Doc {records_processed} Pass {p_idx}"
            }
            
            # Apply all three chunking strategies
            fixed_chunks = fixed_chunker.chunk(passage_text, f"{doc_id}_p{p_idx}", doc_metadata)
            semantic_chunks = semantic_chunker.chunk(passage_text, f"{doc_id}_p{p_idx}", doc_metadata)
            meta_chunks = meta_chunker.chunk(passage_text, f"{doc_id}_p{p_idx}", doc_metadata)
            
            all_chunks = fixed_chunks + semantic_chunks + meta_chunks
            
            for chunk in all_chunks:
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]))
                chunk_text = chunk["text"]
                strategy = chunk["strategy"]
                
                # Format text for E5 embeddings
                emb_text = chunk_text
                if "e5" in settings.EMBEDDING_MODEL_NAME.lower():
                    emb_text = f"passage: {chunk_text}"
                
                # Embed the chunk
                vector = model.encode(emb_text).tolist()
                
                # Create PointStruct
                points_to_upload.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload={
                            "text": chunk_text,
                            "strategy": strategy,
                            "metadata": chunk["metadata"]
                        }
                    )
                )
                
        records_processed += 1
        logger.info(f"Processed dataset row {records_processed}/{limit} ({len(points_to_upload)} total chunks compiled)")

    # Upload to Qdrant in batches
    batch_size = 100
    logger.info(f"Uploading {len(points_to_upload)} points to Qdrant collection '{COLLECTION_NAME}' in batches of {batch_size}...")
    
    for i in range(0, len(points_to_upload), batch_size):
        batch = points_to_upload[i:i + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=batch
        )
        logger.info(f"Uploaded batch {i // batch_size + 1}/{(len(points_to_upload) - 1) // batch_size + 1}")
        
    logger.info("Ingestion completed successfully!")

if __name__ == "__main__":
    limit_num = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    lang_code = sys.argv[2] if len(sys.argv) > 2 else "hi"
    ingest_dataset(limit=limit_num, lang=lang_code)
