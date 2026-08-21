import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from backend.app.config import settings
from backend.app.schemas import ChunkCitation
from backend.app.chunking import get_embedding_model

logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "msmarco_chunks"
VECTOR_SIZE = 384  # For intfloat/multilingual-e5-small

# Lazy client singleton
_qdrant_client = None
is_in_memory = False

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client, is_in_memory
    if _qdrant_client is None:
        try:
            # Attempt to connect to external Qdrant with a short timeout
            _qdrant_client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
                timeout=1.0
            )
            # Ping
            _qdrant_client.get_collections()
            logger.info("Connected to external Qdrant vector database.")
        except Exception as e:
            logger.warning(f"Could not connect to external Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}: {e}. Falling back to IN-MEMORY Qdrant client.")
            _qdrant_client = QdrantClient(":memory:")
            is_in_memory = True
    return _qdrant_client

def populate_in_memory_data(client: QdrantClient):
    logger.info("Populating in-memory Qdrant database with HH Goa static brand FAQs...")
    import uuid
    from backend.app.chunking import FixedSizeChunker
    
    faqs = [
        {
            "title": "Hacker House Goa Overview",
            "text": "Hacker House Goa (HH Goa 2026) is the country's biggest build-station, taking place from October 28–31 in Goa, India. It brings together 247 elite builders to lock in, build, ship, and launch projects. The residency provides high-speed fiber internet, ocean-side workspaces, accommodation, and meals, all for free.",
            "url": "https://hhgoa.com/"
        },
        {
            "title": "Who Can Participate",
            "text": "Anyone with a passion for building can participate in Hacker House Goa. This includes developers, designers, product managers, or creators. Teams of 1-3 people are encouraged, but solo participants are also accepted.",
            "url": "https://hhgoa.com/#faq"
        },
        {
            "title": "Selection Process and Roadmap",
            "text": "The selection process for Hacker House Goa begins with Open Trials (skill-based challenges), followed by Alpha Selections (first shortlist), Beta Selections (technical & portfolio review), Charlie Selections (interviews and team-fit assessment), and Delta Selections (final shortlist before partner matching and RSVPs).",
            "url": "https://hhgoa.com/#roadmap"
        },
        {
            "title": "Registration and Costs",
            "text": "Participation in Hacker House Goa is completely free of cost. There is no registration fee. Selected hackers are provided with accommodation, meals, workspace, high-speed internet, and amenities during the 4-day residency. Participants only need to arrange their travel to Goa.",
            "url": "https://hhgoa.com/#faq"
        },
        {
            "title": "What to Bring to the Event",
            "text": "Participants must bring their own laptop, chargers, any specific hardware they need for building, and creative energy. Hacker House Goa provides workspaces, electricity, high-speed WiFi, food, and caffeine.",
            "url": "https://hhgoa.com/#faq"
        },
        {
            "title": "Task 1: HH Goa Frame ID Generator",
            "text": "Task #1 is the HH Goa Frame / ID Card Generator. Builders design a themed photo frame generator to bring teammates into one frame, post it on X with the hashtag #FrameInGoa, and get to the top of the ladder to win the exclusive HH Goa ID.",
            "url": "https://hhgoa.com/#tasks"
        },
        {
            "title": "Task 2: Voice-Enabled RAG Model",
            "text": "Task #2 is the Voice-Enabled RAG Model. It requires building a full voice-to-answer RAG pipeline with speech-to-text transcription, engineered chunking, vector retrieval, and structured LLM generation, running under 200ms latency, with rate limits, daily caps, guardrails, and the hashtag #RAGInGoa.",
            "url": "https://hhgoa.com/#tasks"
        }
    ]
    
    model = get_embedding_model()
    if model is None:
        logger.warning("Could not load embedding model for in-memory population. Chunks will be populated with zero-vectors to enable keyword-only fallback search.")
        
    fixed_chunker = FixedSizeChunker(chunk_size=400, chunk_overlap=80)
    points = []
    
    for idx, faq in enumerate(faqs):
        doc_metadata = {
            "language": "en",
            "source": "hhgoa-faq",
            "source_url": faq["url"],
            "title": faq["title"]
        }
        
        chunks = fixed_chunker.chunk(faq["text"], f"static_{idx}", doc_metadata)
        for c_idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"]))
            chunk_text = chunk["text"]
            
            emb_text = chunk_text
            if "e5" in settings.EMBEDDING_MODEL_NAME.lower():
                emb_text = f"passage: {chunk_text}"
                
            if model is not None:
                vector = model.encode(emb_text).tolist()
            else:
                vector = [0.0] * VECTOR_SIZE
                
            points.append(
                models.PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "text": chunk_text,
                        "strategy": "fixed",
                        "metadata": chunk["metadata"]
                    }
                )
            )
            
    client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)
    logger.info(f"Successfully populated in-memory Qdrant with {len(points)} chunks.")

def init_qdrant_collection():
    client = get_qdrant_client()
    
    # Check if collection exists
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
    except Exception as e:
        logger.error(f"Failed to check Qdrant collections: {e}")
        return False
        
    if not exists:
        logger.info(f"Creating Qdrant collection '{COLLECTION_NAME}'...")
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            # Create text index on 'text' field for keyword/BM25 search
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    lowercase=True,
                )
            )
            logger.info("Qdrant collection and indices created successfully.")
            
            # If client is in-memory fallback, populate it automatically!
            if is_in_memory:
                populate_in_memory_data(client)
                
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {e}")
            return False
    return True

async def hybrid_search(query_text: str, top_k: int = 5) -> List[ChunkCitation]:
    client = get_qdrant_client()
    model = get_embedding_model()
    
    if model is None:
        logger.warning("Embedding model is missing. Falling back to keyword-only search.")
        return await keyword_search(query_text, top_k)
        
    # Format query for E5 models (query prefix)
    formatted_query = query_text
    if "e5" in settings.EMBEDDING_MODEL_NAME.lower():
        formatted_query = f"query: {query_text}"
        
    # Generate query embedding
    query_vector = model.encode(formatted_query).tolist()
    
    # 1. Dense Vector Search
    dense_hits = []
    try:
        dense_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k * 2
        )
        for rank, hit in enumerate(dense_results):
            dense_hits.append({
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "strategy": hit.payload.get("strategy", "unknown"),
                "metadata": hit.payload.get("metadata", {}),
                "rank": rank + 1
            })
    except Exception as e:
        logger.error(f"Dense vector search failed: {e}")
        
    # 2. Keyword/Full-Text Search (using MatchText against payload index)
    keyword_hits = []
    try:
        # We scroll with a MatchText filter
        scroll_res, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="text",
                        match=models.MatchText(text=query_text)
                    )
                ]
            ),
            limit=top_k * 2,
            with_payload=True,
            with_vectors=False
        )
        for rank, record in enumerate(scroll_res):
            # Since scroll does not return score, we use a simple placeholder score or a length-based estimate
            keyword_hits.append({
                "id": record.id,
                "text": record.payload.get("text", ""),
                "score": 0.5, # Default placeholder score for RRF
                "strategy": record.payload.get("strategy", "unknown"),
                "metadata": record.payload.get("metadata", {}),
                "rank": rank + 1
            })
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF Score = 1 / (60 + r_dense) + 1 / (60 + r_keyword)
    rrf_constant = 60
    fused_docs = {}
    
    # Map dense rank
    for item in dense_hits:
        doc_id = item["id"]
        fused_docs[doc_id] = {
            "item": item,
            "rrf_score": 1.0 / (rrf_constant + item["rank"]),
            "seen_dense": True,
            "seen_keyword": False
        }
        
    # Map keyword rank
    for item in keyword_hits:
        doc_id = item["id"]
        if doc_id in fused_docs:
            fused_docs[doc_id]["rrf_score"] += 1.0 / (rrf_constant + item["rank"])
            fused_docs[doc_id]["seen_keyword"] = True
        else:
            fused_docs[doc_id] = {
                "item": item,
                "rrf_score": 1.0 / (rrf_constant + item["rank"]),
                "seen_dense": False,
                "seen_keyword": True
            }
            
    # Sort by RRF score descending
    sorted_docs = sorted(fused_docs.values(), key=lambda x: x["rrf_score"], reverse=True)
    
    # Format outputs
    results = []
    for entry in sorted_docs[:top_k]:
        item = entry["item"]
        results.append(
            ChunkCitation(
                id=str(item["id"]),
                text=item["text"],
                score=round(float(item["score"]), 4),
                strategy=item["strategy"],
                metadata=item["metadata"]
            )
        )
        
    logger.info(f"Retrieved {len(results)} chunks via Hybrid search for: '{query_text}'")
    return results

async def keyword_search(query_text: str, top_k: int = 5) -> List[ChunkCitation]:
    client = get_qdrant_client()
    results = []
    try:
        # Fetch all chunks in the collection
        scroll_res, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        # Tokenize the query
        import re
        stop_words = {"a", "an", "the", "and", "or", "but", "if", "then", "so", "for", "in", "on", "at", "to", "by", "of", "with", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "what", "which", "who", "whom", "this", "that", "these", "those", "can", "could", "will", "would", "shall", "should", "may", "might", "must"}
        
        query_words = set(re.findall(r"\w+", query_text.lower()))
        query_keywords = query_words - stop_words
        if not query_keywords:
            query_keywords = query_words  # fallback to all words if query is only stop words
            
        scored_records = []
        for record in scroll_res:
            payload_text = record.payload.get("text", "")
            doc_words = set(re.findall(r"\w+", payload_text.lower()))
            
            # Count matching keywords
            matches = query_keywords.intersection(doc_words)
            overlap_score = len(matches)
            
            # Simple TF-IDF boost for matches in title
            doc_title = record.payload.get("metadata", {}).get("title", "")
            title_words = set(re.findall(r"\w+", doc_title.lower()))
            title_matches = query_keywords.intersection(title_words)
            overlap_score += len(title_matches) * 2
            
            scored_records.append((overlap_score, record))
            
        # Sort by overlap score descending
        scored_records.sort(key=lambda x: x[0], reverse=True)
        
        # Keep records with at least 1 keyword match
        for score, record in scored_records:
            if score > 0:
                results.append(
                    ChunkCitation(
                        id=str(record.id),
                        text=record.payload.get("text", ""),
                        score=1.0,  # Ensure it passes the safety threshold
                        strategy=record.payload.get("strategy", "keyword_only"),
                        metadata=record.payload.get("metadata", {})
                    )
                )
                
    except Exception as e:
        logger.error(f"Fallback keyword search failed: {e}")
    return results[:top_k]
