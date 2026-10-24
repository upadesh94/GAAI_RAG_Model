"""Vector store and retrieval management using ChromaDB.

Provides session-isolated vector indexes for resumes, similarity search with
metadata preservation, safe deletion of stored data, and multi-resume isolation.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)


def sanitize_collection_name(name: str) -> str:
    """Sanitize collection name to conform to ChromaDB naming requirements:

    Must be 3-63 chars, contain only letters, numbers, underscores, hyphens,
    and start/end with an alphanumeric character.
    """
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    clean = clean.strip("_-")
    if len(clean) < 3:
        clean = f"coll_{clean}_{uuid.uuid4().hex[:6]}"
    elif len(clean) > 63:
        clean = clean[:63].rstrip("_-")
    return clean


class VectorStoreError(Exception):
    """Raised when ChromaDB operations fail."""


class ResumeVectorStore:
    """Manages isolated ChromaDB vector collections for candidate resumes."""

    def __init__(
        self,
        embedding_model: GeminiEmbeddings,
        persist_directory: Optional[str] = ".chromadb",
        session_id: Optional[str] = None,
    ):
        """Initialize the vector store for a specific session.

        Args:
            embedding_model: Initialized GeminiEmbeddings adapter.
            persist_directory: Directory for persistent ChromaDB storage. If None, in-memory.
            session_id: Unique identifier for session isolation. Generated if None.
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.session_id = session_id or uuid.uuid4().hex
        self.collection_name = sanitize_collection_name(f"doc_{self.session_id}")

        self._vector_db: Optional[Chroma] = None
        self._doc_count: int = 0

    @property
    def doc_count(self) -> int:
        """Return the number of documents indexed in the current collection."""
        return self._doc_count

    def index_documents(self, documents: List[Document]) -> int:
        """Index resume text chunks into the session's isolated ChromaDB collection.

        Args:
            documents: List of chunk Document objects.

        Returns:
            The number of chunks successfully indexed.

        Raises:
            VectorStoreError: If indexing fails or documents list is empty.
        """
        if not documents:
            raise VectorStoreError("Cannot index an empty document list.")

        try:
            # Clear previous collection for this session if it existed
            self.clear_data()

            self._vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
            )
            self._doc_count = len(documents)
            logger.info(
                "Indexed %d chunks into collection '%s'.",
                self._doc_count,
                self.collection_name,
            )
            return self._doc_count

        except Exception as e:
            logger.error("Failed to index documents into ChromaDB: %s", e)
            raise VectorStoreError(f"Vector database indexing failed: {e}") from e

    def retrieve(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        """Retrieve the k most relevant chunks for the user's question.

        Args:
            query: Question or search string.
            k: Number of relevant chunks to retrieve.

        Returns:
            List of retrieved Document objects with page and source metadata.

        Raises:
            VectorStoreError: If vector store has not been indexed yet.
        """
        if not self._vector_db:
            raise VectorStoreError("No resume has been indexed yet. Please upload a resume first.")

        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        try:
            retriever = self._vector_db.as_retriever(search_kwargs={"k": k})
            results = retriever.invoke(cleaned_query)
            return results
        except Exception as e:
            logger.error("Retrieval failed for query '%s': %s", cleaned_query, e)
            raise VectorStoreError(f"Vector search retrieval failed: {e}") from e

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Retrieve top k chunks along with their relevance scores/distances.

        Args:
            query: Question string.
            k: Number of chunks.

        Returns:
            List of (Document, score) tuples.
        """
        if not self._vector_db:
            raise VectorStoreError("No resume has been indexed yet. Please upload a resume first.")

        cleaned_query = (query or "").strip()
        if not cleaned_query:
            return []

        try:
            results = self._vector_db.similarity_search_with_score(cleaned_query, k=k)
            return results
        except Exception as e:
            logger.warning("Score retrieval failed, falling back to basic retrieve: %s", e)
            basic_docs = self.retrieve(cleaned_query, k=k)
            return [(doc, 0.0) for doc in basic_docs]

    def clear_data(self) -> None:
        """Safely delete the current session's collection and free storage."""
        if self._vector_db is not None:
            try:
                self._vector_db.delete_collection()
                logger.info("Deleted ChromaDB collection '%s'.", self.collection_name)
            except Exception as e:
                logger.warning("Notice on deleting collection '%s': %s", self.collection_name, e)
            finally:
                self._vector_db = None
                self._doc_count = 0
