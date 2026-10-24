"""Unit tests for ResumeQAService, context grounding, and fallback behavior."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from doc_qa.answering import (
    FALLBACK_NOT_AVAILABLE,
    SYSTEM_INSTRUCTION,
    QAResponse,
    ResumeQAService,
)
from doc_qa.config import AppConfig
from doc_qa.vector_store import ResumeVectorStore


class MockVectorStore:
    """Mock vector store for controlled retrieval unit testing."""

    def __init__(self, documents=None):
        self.documents = documents or []

    def retrieve_with_scores(self, query: str, k: int = 4):
        if not self.documents:
            return []
        return [(doc, 0.1) for doc in self.documents[:k]]


class TestResumeQAService:
    """Tests for QA service grounding, fallbacks, and error handling."""

    def test_missing_api_configuration_raises_error(self):
        vstore = MockVectorStore()
        with pytest.raises(ValueError, match="Either an active genai.Client or a valid api_key"):
            ResumeQAService(vector_store=vstore, client=None, api_key=None)

    def test_empty_question_returns_friendly_message(self):
        vstore = MockVectorStore()
        mock_client = MagicMock()
        service = ResumeQAService(vector_store=vstore, client=mock_client)

        resp = service.answer_question("")
        assert "Please enter a question" in resp.answer
        assert not resp.is_available
        assert len(resp.evidence) == 0

    def test_empty_retrieval_returns_fallback_without_calling_gemini(self):
        vstore = MockVectorStore(documents=[])
        mock_client = MagicMock()
        service = ResumeQAService(vector_store=vstore, client=mock_client)

        resp = service.answer_question("What is the candidate's salary?")
        assert resp.answer == FALLBACK_NOT_AVAILABLE
        assert not resp.is_available
        mock_client.models.generate_content.assert_not_called()

    def test_grounded_answer_with_evidence(self):
        sample_doc = Document(
            page_content="Education: B.Tech in Computer Science from Stanford University.",
            metadata={"page": 1, "source": "resume.pdf", "chunk_id": 1},
        )
        vstore = MockVectorStore(documents=[sample_doc])

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The candidate holds a B.Tech in Computer Science from Stanford University."
        mock_client.models.generate_content.return_value = mock_response

        service = ResumeQAService(
            vector_store=vstore,
            client=mock_client,
            model_name="gemini-3.5-flash",
        )

        resp = service.answer_question("What is the candidate's degree?")

        assert resp.is_available is True
        assert "Stanford University" in resp.answer
        assert len(resp.evidence) == 1
        assert resp.evidence[0].page == 1
        assert "Stanford" in resp.evidence[0].excerpt

        # Verify prompt contained grounding instructions and context
        call_args = mock_client.models.generate_content.call_args
        prompt_sent = call_args.kwargs["contents"]
        assert "Resume Context:" in prompt_sent
        assert "Stanford University" in prompt_sent
        assert SYSTEM_INSTRUCTION in prompt_sent

    def test_missing_information_returns_fallback(self):
        sample_doc = Document(
            page_content="Skills: HTML, CSS, JavaScript.",
            metadata={"page": 1, "source": "frontend.pdf"},
        )
        vstore = MockVectorStore(documents=[sample_doc])

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = FALLBACK_NOT_AVAILABLE
        mock_client.models.generate_content.return_value = mock_response

        service = ResumeQAService(vector_store=vstore, client=mock_client)
        resp = service.answer_question("Does the candidate have Kubernetes experience?")

        assert resp.answer == FALLBACK_NOT_AVAILABLE
        assert resp.is_available is False

    def test_external_api_failure_handled_gracefully(self):
        sample_doc = Document(
            page_content="Skills: Python.",
            metadata={"page": 1, "source": "resume.pdf"},
        )
        vstore = MockVectorStore(documents=[sample_doc])

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API quota exceeded")

        service = ResumeQAService(vector_store=vstore, client=mock_client)
        resp = service.answer_question("What skills are listed?")

        assert "AI service error" in resp.answer
        assert not resp.is_available
        assert resp.error is not None
