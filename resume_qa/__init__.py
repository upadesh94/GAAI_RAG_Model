"""AI Resume Question Answering System (RAG).

A modular Python package providing PDF text extraction, Gemini embeddings adapter,
ChromaDB vector store isolation, and context-grounded question answering.
"""

from .config import (
    AppConfig,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_RETRIEVAL_K,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_GENERATION_MODELS,
    get_default_config,
)
from .pdf_processing import (
    EmptyPDFError,
    EncryptedPDFError,
    PDFProcessingError,
    PDFValidationError,
    ScannedPDFError,
    extract_text_from_pdf,
    split_documents_into_chunks,
    validate_pdf_bytes,
)
from .embeddings import GeminiEmbeddings, GeminiEmbeddingsError
from .vector_store import ResumeVectorStore, VectorStoreError
from .answering import (
    FALLBACK_NOT_AVAILABLE,
    SYSTEM_INSTRUCTION,
    EvidenceItem,
    QAResponse,
    ResumeQAService,
)

__all__ = [
    "AppConfig",
    "get_default_config",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_GENERATION_MODEL",
    "DEFAULT_RETRIEVAL_K",
    "SUPPORTED_EMBEDDING_MODELS",
    "SUPPORTED_GENERATION_MODELS",
    "extract_text_from_pdf",
    "split_documents_into_chunks",
    "validate_pdf_bytes",
    "PDFProcessingError",
    "PDFValidationError",
    "EncryptedPDFError",
    "ScannedPDFError",
    "EmptyPDFError",
    "GeminiEmbeddings",
    "GeminiEmbeddingsError",
    "ResumeVectorStore",
    "VectorStoreError",
    "ResumeQAService",
    "QAResponse",
    "EvidenceItem",
    "FALLBACK_NOT_AVAILABLE",
    "SYSTEM_INSTRUCTION",
]
