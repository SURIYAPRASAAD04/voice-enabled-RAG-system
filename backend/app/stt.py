import httpx
import logging
from abc import ABC, abstractmethod
from backend.app.config import settings

logger = logging.getLogger(__name__)

class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, filename: str = "query.wav", content_type: str = "audio/wav") -> str:
        """Transcribe audio bytes to text."""
        pass

class SarvamSTTProvider(STTProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.sarvam.ai/speech-to-text"

    async def transcribe(self, audio_bytes: bytes, filename: str = "query.wav", content_type: str = "audio/wav") -> str:
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY is not set.")
        
        headers = {
            "api-subscription-key": self.api_key
        }
        
        # Sarvam speech-to-text uses "saaras:v3" and mode "codemix" or "transcribe". 
        # Using "codemix" is highly recommended for Indic + English mix queries.
        files = {
            "file": (filename, audio_bytes, content_type)
        }
        data = {
            "model": "saaras:v3",
            "mode": "codemix"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            logger.info("Calling Sarvam AI STT API...")
            response = await client.post(self.url, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                logger.error(f"Sarvam STT failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"Sarvam STT failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            # The structure returned by Sarvam is typically {"transcript": "..."}
            transcript = result.get("transcript", "").strip()
            logger.info(f"Sarvam AI STT transcription success: '{transcript}'")
            return transcript

class ElevenLabsSTTProvider(STTProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.elevenlabs.io/v1/speech-to-text"

    async def transcribe(self, audio_bytes: bytes, filename: str = "query.wav", content_type: str = "audio/wav") -> str:
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set.")
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        files = {
            "file": (filename, audio_bytes, content_type)
        }
        data = {
            "model_id": "scribe_v2"
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            logger.info("Calling ElevenLabs STT API...")
            response = await client.post(self.url, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                logger.error(f"ElevenLabs STT failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"ElevenLabs STT failed with status {response.status_code}: {response.text}")
            
            result = response.json()
            # The structure returned by ElevenLabs STT is typically {"text": "..."}
            transcript = result.get("text", "").strip()
            logger.info(f"ElevenLabs STT transcription success: '{transcript}'")
            return transcript

class DummySTTProvider(STTProvider):
    """Fallback / Mock STT provider for offline testing or missing keys."""
    async def transcribe(self, audio_bytes: bytes, filename: str = "query.wav", content_type: str = "audio/wav") -> str:
        logger.warning("Using DummySTTProvider. Returning dummy transcript.")
        return "Task 2 Submission: Voice-Enabled RAG Model"

def get_stt_provider() -> STTProvider:
    provider_name = settings.STT_PROVIDER.lower()
    
    if provider_name == "sarvam":
        if settings.SARVAM_API_KEY:
            return SarvamSTTProvider(settings.SARVAM_API_KEY)
        else:
            logger.warning("Sarvam API key missing. Falling back to Dummy STT.")
            return DummySTTProvider()
    elif provider_name == "elevenlabs":
        if settings.ELEVENLABS_API_KEY:
            return ElevenLabsSTTProvider(settings.ELEVENLABS_API_KEY)
        else:
            logger.warning("ElevenLabs API key missing. Falling back to Dummy STT.")
            return DummySTTProvider()
    else:
        logger.warning(f"Unknown STT provider '{provider_name}'. Falling back to Dummy STT.")
        return DummySTTProvider()
