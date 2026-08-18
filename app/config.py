"""Application configuration loaded from environment variables via pydantic-settings.

No hardcoded model names, paths, or credentials live in the source code.
All runtime values are resolved from ``.env`` (or the process environment).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorDbType = Literal["chroma", "qdrant"]


class Settings(BaseSettings):
    """Typed access to every configurable value used by the service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Credentials
    openai_api_key: SecretStr = Field(..., description="OpenAI API key.")
    llama_cloud_api_key: SecretStr = Field(..., description="LlamaCloud / LlamaParse API key.")

    # Model configuration
    llm_model: str = Field(default="gpt-4o-mini", description="LLM used for answer generation.")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model for chunk vectorization."
    )

    # Vector store configuration
    vector_db_type: VectorDbType = Field(default="chroma", description="Backend vector database.")
    chroma_persist_dir: str = Field(
        default="data/chroma", description="ChromaDB persistence directory."
    )
    collection_name: str = Field(
        default="multimodal_docs", description="Collection/namespace name."
    )
    qdrant_url: str | None = Field(
        default=None, description="Qdrant Cloud URL (if VECTOR_DB_TYPE=qdrant)."
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None, description="Qdrant Cloud API key (if Qdrant)."
    )

    # Chunking
    chunk_size: int = Field(default=1500, ge=100, description="Target chunk length in characters.")
    chunk_overlap: int = Field(default=150, ge=0, description="Overlap between text chunks.")

    # Retrieval / generation
    top_k: int = Field(default=5, ge=1, le=50, description="Default chunks retrieved per query.")
    llm_temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="LLM sampling temperature."
    )
    max_retries: int = Field(default=3, ge=1, description="Retry count for external API calls.")

    @property
    def persist_dir(self) -> Path:
        """Absolute ChromaDB persistence directory, created on demand."""
        path = Path(self.chroma_persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
