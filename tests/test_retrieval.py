"""Unit tests for ResumeVectorStore and session isolation."""

import hashlib
import pytest
from langchain_core.documents import Document

from doc_qa.embeddings import GeminiEmbeddings
from doc_qa.vector_store import ResumeVectorStore, VectorStoreError


class DeterministicMockEmbeddings:
    """Mock embeddings that return deterministic 64-dimensional vectors for testing."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        # Hash text to create a deterministic vector
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat hash to fill dimensions
        floats = []
        for i in range(self.dim):
            byte_val = h[i % len(h)]
            floats.append(float(byte_val) / 255.0)
        return floats


class TestResumeVectorStore:
    """Tests for vector store indexing, search, and session isolation."""

    @pytest.fixture
    def mock_embeddings(self):
        return DeterministicMockEmbeddings()

    def test_indexing_and_retrieval(self, mock_embeddings, tmp_path):
        store = ResumeVectorStore(
            embedding_model=mock_embeddings,
            persist_directory=str(tmp_path / "chroma_test1"),
            session_id="session_alpha",
        )

        docs = [
            Document(
                page_content="Candidate has 5 years of experience in Python and Django.",
                metadata={"page": 1, "source": "resume.pdf", "chunk_id": 1},
            ),
            Document(
                page_content="Education: Bachelor of Science in Computer Science from Berkeley.",
                metadata={"page": 2, "source": "resume.pdf", "chunk_id": 2},
            ),
        ]

        indexed_count = store.index_documents(docs)
        assert indexed_count == 2
        assert store.doc_count == 2

        results = store.retrieve("Python programming experience", k=2)
        assert len(results) > 0
        assert any("Python" in d.page_content for d in results)
        assert results[0].metadata["page"] in (1, 2)

    def test_session_isolation_prevents_cross_resume_contamination(
        self, mock_embeddings, tmp_path
    ):
        persist_dir = str(tmp_path / "chroma_shared")

        # Session 1: Candidate A (John Doe)
        store_a = ResumeVectorStore(
            embedding_model=mock_embeddings,
            persist_directory=persist_dir,
            session_id="candidate_a",
        )
        docs_a = [
            Document(
                page_content="John Doe is a Senior Rust and C++ Systems Engineer.",
                metadata={"page": 1, "source": "john_doe.pdf"},
            )
        ]
        store_a.index_documents(docs_a)

        # Session 2: Candidate B (Jane Smith)
        store_b = ResumeVectorStore(
            embedding_model=mock_embeddings,
            persist_directory=persist_dir,
            session_id="candidate_b",
        )
        docs_b = [
            Document(
                page_content="Jane Smith is an expert in Graphic Design and Figma.",
                metadata={"page": 1, "source": "jane_smith.pdf"},
            )
        ]
        store_b.index_documents(docs_b)

        # Candidate A should only retrieve John Doe
        retrieved_a = store_a.retrieve("What are the candidate's skills?", k=5)
        for doc in retrieved_a:
            assert "John Doe" in doc.page_content
            assert "Jane Smith" not in doc.page_content

        # Candidate B should only retrieve Jane Smith
        retrieved_b = store_b.retrieve("What are the candidate's skills?", k=5)
        for doc in retrieved_b:
            assert "Jane Smith" in doc.page_content
            assert "John Doe" not in doc.page_content

        # Cleanup
        store_a.clear_data()
        store_b.clear_data()

    def test_retrieve_without_indexing_raises_error(self, mock_embeddings, tmp_path):
        store = ResumeVectorStore(
            embedding_model=mock_embeddings,
            persist_directory=str(tmp_path / "chroma_empty"),
            session_id="unindexed_session",
        )
        with pytest.raises(VectorStoreError, match="No resume has been indexed"):
            store.retrieve("Where did the candidate study?")

    def test_index_empty_documents_raises_error(self, mock_embeddings, tmp_path):
        store = ResumeVectorStore(
            embedding_model=mock_embeddings,
            persist_directory=str(tmp_path / "chroma_empty2"),
            session_id="empty_docs_session",
        )
        with pytest.raises(VectorStoreError, match="Cannot index an empty document list"):
            store.index_documents([])

    def test_clear_data_resets_store(self, mock_embeddings, tmp_path):
        store = ResumeVectorStore(
            embedding_model=mock_embeddings,
            persist_directory=str(tmp_path / "chroma_clear"),
            session_id="clear_test_session",
        )
        docs = [
            Document(
                page_content="Certification: AWS Solutions Architect Associate.",
                metadata={"page": 1, "source": "cert.pdf"},
            )
        ]
        store.index_documents(docs)
        assert store.doc_count == 1

        store.clear_data()
        assert store.doc_count == 0
        with pytest.raises(VectorStoreError):
            store.retrieve("AWS cert")
