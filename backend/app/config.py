import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General
    ENV: str = "development"
    
    # STT Settings
    STT_PROVIDER: str = "sarvam"  # "sarvam" | "elevenlabs"
    SARVAM_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    
    # LLM Settings
    LLM_PROVIDER: str = "groq"  # "groq" | "openai" | "anthropic"
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    # LLM Model overrides
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
    
    # Vector DB
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str = ""
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Rate Limiting & Caps
    DAILY_REQUEST_CAP: int = 200
    RATE_LIMIT_PER_MIN: int = 10
    MAX_AUDIO_DURATION_SECONDS: int = 30
    
    # RAG Settings
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    DISABLE_LOCAL_EMBEDDINGS: bool = False
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

settings = Settings()
