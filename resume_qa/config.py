"""Configuration management for the AI Resume Question Answering System.

Provides centralized, configurable parameters for Gemini models, chunking,
retrieval, file upload limits, and safe API key handling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Automatically load environment variables from .env if present
load_dotenv()

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_GENERATION_MODEL = "gemini-3.5-flash-lite"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 4
DEFAULT_MAX_FILE_SIZE_MB = 10
DEFAULT_CHROMA_DIR = ".chromadb"

# Supported generation models in priority order for UI or fallback
SUPPORTED_GENERATION_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

SUPPORTED_EMBEDDING_MODELS = [
    "gemini-embedding-2",
    "gemini-embedding-001",
]


@dataclass
class AppConfig:
    """Application configuration with runtime validation and safe display."""

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
    )
    generation_model: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_GENERATION_MODEL", DEFAULT_GENERATION_MODEL
        )
    )
    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))
        )
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))
        )
    )
    retrieval_k: int = field(
        default_factory=lambda: int(
            os.getenv("RETRIEVAL_K", str(DEFAULT_RETRIEVAL_K))
        )
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR
        )
    )
    max_file_size_mb: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_FILE_SIZE_MB", str(DEFAULT_MAX_FILE_SIZE_MB))
        )
    )

    def validate(self) -> None:
        """Validate critical configuration fields."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is missing. Please set it in your environment, "
                ".env file, or the application interface."
            )
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be a positive integer.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE.")
        if self.retrieval_k <= 0:
            raise ValueError("RETRIEVAL_K must be greater than 0.")

    def get_masked_api_key(self) -> str:
        """Return a safely masked representation of the API key for UI display."""
        if not self.api_key:
            return "Not configured"
        key = self.api_key.strip()
        if len(key) <= 8:
            return "******"
        return f"{key[:4]}...{key[-4:]}"


def get_default_config() -> AppConfig:
    """Return an AppConfig instance populated from environment."""
    return AppConfig()
