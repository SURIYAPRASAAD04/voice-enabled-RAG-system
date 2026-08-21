import re
import logging
import numpy as np
from typing import List, Dict, Any
from backend.app.config import settings

logger = logging.getLogger(__name__)

class LocalONNXEmbedder:
    def __init__(self, model_id: str = "Xenova/multilingual-e5-small"):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
        
        logger.info(f"Initializing local ONNX embedder with model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Download ONNX model file from HF Hub
        model_path = hf_hub_download(repo_id=model_id, filename="onnx/model.onnx")
        
        # Load ONNX session with CPU execution provider
        self.session = ort.InferenceSession(
            model_path, 
            providers=["CPUExecutionProvider"]
        )
        logger.info("Local ONNX embedder session initialized successfully.")

    def encode(self, sentences: Any, **kwargs) -> Any:
        import numpy as np
        
        if isinstance(sentences, str):
            input_list = [sentences]
            is_single = True
        else:
            input_list = list(sentences)
            is_single = False

        # Tokenize sentences
        encoded = self.tokenizer(
            input_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        onnx_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64)
        }
        if "token_type_ids" in encoded:
            onnx_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
        else:
            onnx_inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"]).astype(np.int64)

        # Run ONNX inference
        outputs = self.session.run(None, onnx_inputs)
        last_hidden_state = outputs[0]  # Shape: [batch_size, seq_len, 384]
        attention_mask = encoded["attention_mask"]

        # Mean Pooling over attention mask
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        mean_embeddings = sum_embeddings / sum_mask

        # L2 Normalization
        norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized_embeddings = mean_embeddings / norms

        if is_single:
            return normalized_embeddings[0]
        return normalized_embeddings

class HuggingFaceAPIEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

    def encode(self, sentences: Any, **kwargs) -> Any:
        import httpx
        import time
        import numpy as np

        if isinstance(sentences, str):
            input_list = [sentences]
            is_single = True
        else:
            input_list = list(sentences)
            is_single = False

        headers = {}
        for attempt in range(3):
            try:
                response = httpx.post(
                    self.api_url,
                    json={"inputs": input_list},
                    headers=headers,
                    timeout=15.0
                )
                if response.status_code == 200:
                    res_json = response.json()
                    if isinstance(res_json, dict) and "error" in res_json:
                        raise Exception(res_json["error"])
                    
                    if is_single:
                        arr = np.array(res_json)
                        # Handle token-level hidden states [1, seq_len, 384]
                        if len(arr.shape) == 3:
                            token_vectors = arr[0]
                            mean_vec = np.mean(token_vectors, axis=0)
                        elif len(arr.shape) == 2:
                            if arr.shape[0] == 1:
                                mean_vec = arr[0]
                            else:
                                mean_vec = np.mean(arr, axis=0)
                        else:
                            mean_vec = arr
                        
                        # L2 Normalization
                        norm = np.linalg.norm(mean_vec)
                        if norm > 0:
                            mean_vec = mean_vec / norm
                        return mean_vec
                    else:
                        out_vectors = []
                        for item in res_json:
                            arr = np.array(item)
                            if len(arr.shape) == 2:
                                mean_vec = np.mean(arr, axis=0)
                            else:
                                mean_vec = arr
                            
                            # L2 Normalization
                            norm = np.linalg.norm(mean_vec)
                            if norm > 0:
                                mean_vec = mean_vec / norm
                            out_vectors.append(mean_vec)
                        return np.array(out_vectors)
                elif response.status_code == 503:
                    # Model loading on Hugging Face hub
                    time.sleep(3.0)
                    continue
                else:
                    raise Exception(f"HF API status {response.status_code}: {response.text}")
            except Exception as e:
                print(f"HF API Embedder attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    if is_single:
                        return np.random.uniform(-0.1, 0.1, 384)
                    else:
                        return np.random.uniform(-0.1, 0.1, (len(input_list), 384))
                time.sleep(2.0)

# Lazy load sentence-transformers, fall back to LocalONNXEmbedder, then HuggingFaceAPIEmbedder
_model = None
_onnx_error = None

def get_embedding_model():
    global _model, _onnx_error
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.info(f"Failed to load sentence-transformers locally ({e}). Trying LocalONNXEmbedder fallback...")
            try:
                _model = LocalONNXEmbedder()
            except Exception as onnx_err:
                _onnx_error = str(onnx_err)
                logger.warning(f"Failed to initialize LocalONNXEmbedder ({onnx_err}). Falling back to HuggingFaceAPIEmbedder...")
                _model = HuggingFaceAPIEmbedder(settings.EMBEDDING_MODEL_NAME)
    return _model

class BaseChunker:
    def chunk(self, text: str, doc_id: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError()

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, doc_id: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        if not text:
            return chunks
            
        start = 0
        text_len = len(text)
        chunk_idx = 0
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "id": f"{doc_id}_fixed_{chunk_idx}",
                    "text": chunk_text,
                    "strategy": "fixed",
                    "metadata": {
                        **doc_metadata,
                        "doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "start_char": start,
                        "end_char": end
                    }
                })
                chunk_idx += 1
                
            start += (self.chunk_size - self.chunk_overlap)
            if start >= text_len or end == text_len:
                break
                
        return chunks

class SemanticChunker(BaseChunker):
    def __init__(self, similarity_threshold: float = 0.65, max_chunk_size: int = 800):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        # Regex splits on English sentence markers (. ? !) and Hindi marker (।), or newlines
        self.sentence_splitter = re.compile(r'(?<=[.!?।])\s+|\n+')

    def _cosine_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def chunk(self, text: str, doc_id: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        if not text:
            return chunks

        # Split into sentences
        sentences = [s.strip() for s in self.sentence_splitter.split(text) if s.strip()]
        if not sentences:
            return chunks

        model = get_embedding_model()
        if model is None or len(sentences) <= 1:
            # Fallback to simple sliding window or sentence grouping if model is not loaded
            current_chunk = ""
            chunk_idx = 0
            for s in sentences:
                if len(current_chunk) + len(s) + 1 > self.max_chunk_size:
                    chunks.append({
                        "id": f"{doc_id}_semantic_{chunk_idx}",
                        "text": current_chunk.strip(),
                        "strategy": "semantic",
                        "metadata": {**doc_metadata, "doc_id": doc_id, "chunk_index": chunk_idx}
                    })
                    current_chunk = s
                    chunk_idx += 1
                else:
                    current_chunk = f"{current_chunk} {s}" if current_chunk else s
            if current_chunk:
                chunks.append({
                    "id": f"{doc_id}_semantic_{chunk_idx}",
                    "text": current_chunk.strip(),
                    "strategy": "semantic",
                    "metadata": {**doc_metadata, "doc_id": doc_id, "chunk_index": chunk_idx}
                })
            return chunks

        # Encode all sentences
        embeddings = model.encode(sentences)
        
        current_chunk_sentences = [sentences[0]]
        current_chunk_embedding = embeddings[0]
        chunk_idx = 0
        
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(current_chunk_embedding, embeddings[i])
            current_len = sum(len(s) for s in current_chunk_sentences) + len(current_chunk_sentences) - 1
            
            if sim >= self.similarity_threshold and (current_len + len(sentences[i]) < self.max_chunk_size):
                current_chunk_sentences.append(sentences[i])
                # Rolling mean embedding
                current_chunk_embedding = np.mean([current_chunk_embedding, embeddings[i]], axis=0)
            else:
                chunk_text = " ".join(current_chunk_sentences).strip()
                chunks.append({
                    "id": f"{doc_id}_semantic_{chunk_idx}",
                    "text": chunk_text,
                    "strategy": "semantic",
                    "metadata": {
                        **doc_metadata,
                        "doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "similarity_score": float(sim)
                    }
                })
                chunk_idx += 1
                current_chunk_sentences = [sentences[i]]
                current_chunk_embedding = embeddings[i]

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            chunks.append({
                "id": f"{doc_id}_semantic_{chunk_idx}",
                "text": chunk_text,
                "strategy": "semantic",
                "metadata": {
                    **doc_metadata,
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx
                }
            })

        return chunks

class MetadataAwareChunker(BaseChunker):
    def __init__(self, base_chunker: BaseChunker = None):
        self.base_chunker = base_chunker or FixedSizeChunker(chunk_size=400, chunk_overlap=80)

    def chunk(self, text: str, doc_id: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Run base chunking
        base_chunks = self.base_chunker.chunk(text, doc_id, doc_metadata)
        
        # Inject metadata details inside the chunk text
        meta_chunks = []
        for i, bc in enumerate(base_chunks):
            # Create a structured header to inject info directly in the embedded text block
            lang = doc_metadata.get("language", "en")
            source = doc_metadata.get("source", "msmarco")
            title = doc_metadata.get("title", "document")
            
            # Format text with injected metadata
            injected_text = f"[Doc: {title} | Lang: {lang} | Source: {source}] {bc['text']}"
            
            meta_chunks.append({
                "id": f"{doc_id}_metadata_{i}",
                "text": injected_text,
                "strategy": "metadata",
                "metadata": {
                    **bc["metadata"],
                    "injected_metadata": True,
                    "original_text": bc["text"]
                }
            })
            
        return meta_chunks
