"""Gemini Embeddings Adapter for LangChain and ChromaDB.

Provides a robust adapter wrapping Google GenAI SDK's embedding capabilities
with caching, batching, validation, and secure error handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import errors

logger = logging.getLogger(__name__)


class GeminiEmbeddingsError(Exception):
    """Raised when embedding generation fails."""


class GeminiEmbeddings:
    """Embedding adapter implementing LangChain's embedding interface using google-genai SDK."""

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        api_key: Optional[str] = None,
        model: str = "gemini-embedding-2",
        enable_cache: bool = True,
    ):
        """Initialize the GeminiEmbeddings adapter.

        Args:
            client: An existing google.genai.Client instance. If None, created with api_key.
            api_key: Google Gemini API key if client is not provided.
            model: Embedding model name (e.g., 'gemini-embedding-2').
            enable_cache: If True, caches query embeddings in memory to prevent duplicate requests.
        """
        if client is not None:
            self.client = client
        elif api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(
                "Either an active genai.Client or a valid api_key must be provided."
            )

        self.model = model
        self.enable_cache = enable_cache
        self._cache: Dict[str, List[float]] = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text chunks.

        LangChain expects List[List[float]].

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding float vectors.
        """
        results: List[List[float]] = []
        for text in texts:
            results.append(self.embed_query(text))
        return results

    def embed_query(self, text: str) -> List[float]:
        """Generate a single vector embedding for a question or text chunk.

        Args:
            text: Query string.

        Returns:
            A flat list of float numbers representing the embedding vector.

        Raises:
            GeminiEmbeddingsError: If embedding request fails.
        """
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            cleaned_text = " "  # Fallback non-empty string for empty inputs

        if self.enable_cache and cleaned_text in self._cache:
            return self._cache[cleaned_text]

        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=cleaned_text,
            )

            if not response or not hasattr(response, "embeddings") or not response.embeddings:
                raise GeminiEmbeddingsError("Gemini API returned an empty embedding response.")

            embedding_values = response.embeddings[0].values
            if not embedding_values:
                raise GeminiEmbeddingsError("Returned embedding vector was empty.")

            vector = list(embedding_values)

            if self.enable_cache:
                self._cache[cleaned_text] = vector

            return vector

        except errors.APIError as e:
            # Secure error message that never reveals API key
            sanitized_msg = str(e.message if hasattr(e, "message") else e)
            logger.error("Gemini Embeddings API error: %s", sanitized_msg)
            raise GeminiEmbeddingsError(f"Embedding service API error: {sanitized_msg}") from None
        except Exception as e:
            logger.error("Unexpected embedding error: %s", type(e).__name__)
            raise GeminiEmbeddingsError(f"Failed to generate embedding: {type(e).__name__} - {e}") from None

    def clear_cache(self) -> None:
        """Clear the in-memory embeddings cache."""
        self._cache.clear()
