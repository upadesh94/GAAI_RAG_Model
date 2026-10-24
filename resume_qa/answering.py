"""Grounded Question-Answering Service using Google Gemini.

Enforces strict context grounding, retrieves relevant resume chunks before prompting,
applies the required system instructions, prevents hallucination, and returns
the answer alongside cited page references.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import errors
from langchain_core.documents import Document

from .config import SUPPORTED_GENERATION_MODELS
from .vector_store import ResumeVectorStore

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a document question-answering assistant. Answer using only the information "
    "explicitly supported by the supplied document context. Do not invent facts or details "
    "that are not present in the provided context. If the requested information cannot be found "
    "in the provided context, respond with: Information not available in the document. "
    "If the evidence is ambiguous, explain the uncertainty instead of guessing. "
    "Keep the answer clear and concise."
)

FALLBACK_NOT_AVAILABLE = "Information not available in the document."


@dataclass
class EvidenceItem:
    """Supporting evidence item with page reference and excerpt."""

    page: int
    source: str
    chunk_id: Optional[int]
    excerpt: str
    score: Optional[float] = None


@dataclass
class QAResponse:
    """Structured QA response containing the generated answer and supporting evidence."""

    question: str
    answer: str
    evidence: List[EvidenceItem] = field(default_factory=list)
    model_used: str = ""
    is_available: bool = True
    error: Optional[str] = None


class ResumeQAService:
    """Service orchestrating grounded question answering over resume vector stores."""

    def __init__(
        self,
        vector_store: ResumeVectorStore,
        client: Optional[genai.Client] = None,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.5-flash",
        system_instruction: str = SYSTEM_INSTRUCTION,
    ):
        """Initialize the QA service.

        Args:
            vector_store: An indexed ResumeVectorStore instance.
            client: An existing google.genai.Client instance.
            api_key: API key to create client if client is None.
            model_name: Model identifier to use for generation.
            system_instruction: Grounding instructions for the LLM.
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.system_instruction = system_instruction

        if client is not None:
            self.client = client
        elif api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError("Either an active genai.Client or a valid api_key is required.")

    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a clean context block with page labels."""
        blocks: List[str] = []
        for i, doc in enumerate(documents):
            page_num = doc.metadata.get("page", "Unknown")
            blocks.append(f"[Excerpt {i+1} - Page {page_num}]:\n{doc.page_content}")
        return "\n\n".join(blocks)

    def _call_gemini_with_fallback(self, prompt: str) -> tuple[str, str]:
        """Invoke Gemini with automatic fallback across supported flash models if one fails.

        Returns:
            Tuple of (generated_text, model_name_used).
        """
        # Place primary model first, followed by alternates
        models_to_try = [self.model_name] + [
            m for m in SUPPORTED_GENERATION_MODELS if m != self.model_name
        ]

        last_error = None
        for model in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                text_output = ""
                if hasattr(response, "candidates") and response.candidates:
                    first_cand = response.candidates[0]
                    if hasattr(first_cand, "content") and hasattr(first_cand.content, "parts"):
                        parts_text = [
                            p.text for p in first_cand.content.parts if hasattr(p, "text") and p.text
                        ]
                        text_output = "".join(parts_text).strip()

                if not text_output and hasattr(response, "text") and response.text:
                    text_output = response.text.strip()

                if text_output:
                    return text_output, model
            except (errors.APIError, Exception) as e:
                logger.warning("Generation with model '%s' failed: %s. Trying fallback...", model, e)
                last_error = e
                continue

        raise RuntimeError(f"All Gemini generation models failed. Last error: {last_error}")

    def answer_question(
        self,
        question: str,
        k: int = 4,
    ) -> QAResponse:
        """Answer a natural-language question grounded strictly in the indexed resume.

        Args:
            question: User's question about the resume.
            k: Number of relevant chunks to retrieve.

        Returns:
            QAResponse containing the answer and evidence citations.
        """
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            return QAResponse(
                question="",
                answer="Please enter a question to ask about the document.",
                evidence=[],
                model_used="",
                is_available=False,
            )

        # 1. Retrieve relevant chunks from vector store
        try:
            retrieved_items = self.vector_store.retrieve_with_scores(cleaned_question, k=k)
        except Exception as e:
            logger.error("Error during chunk retrieval: %s", e)
            return QAResponse(
                question=cleaned_question,
                answer=f"Unable to retrieve resume context: {e}",
                evidence=[],
                model_used="",
                is_available=False,
                error=str(e),
            )

        # If no chunks were returned, respond with fallback immediately
        if not retrieved_items:
            return QAResponse(
                question=cleaned_question,
                answer=FALLBACK_NOT_AVAILABLE,
                evidence=[],
                model_used="",
                is_available=False,
            )

        # Build evidence items
        evidence_list: List[EvidenceItem] = []
        retrieved_docs: List[Document] = []
        for doc, score in retrieved_items:
            retrieved_docs.append(doc)
            evidence_list.append(
                EvidenceItem(
                    page=int(doc.metadata.get("page", 1)),
                    source=str(doc.metadata.get("source", "resume.pdf")),
                    chunk_id=doc.metadata.get("chunk_id"),
                    excerpt=doc.page_content,
                    score=float(score) if score is not None else None,
                )
            )

        # 2. Build the grounded prompt
        context_text = self._format_context(retrieved_docs)
        prompt = f"""{self.system_instruction}

Resume Context:
{context_text}

Question:
{cleaned_question}

Answer:"""

        # 3. Call Gemini
        try:
            raw_answer, model_used = self._call_gemini_with_fallback(prompt)
        except Exception as e:
            logger.error("Error generating answer from Gemini: %s", e)
            return QAResponse(
                question=cleaned_question,
                answer=f"Could not generate answer due to an AI service error: {e}",
                evidence=evidence_list,
                model_used="",
                is_available=False,
                error=str(e),
            )

        # Check if the model answered with the fallback
        is_available = FALLBACK_NOT_AVAILABLE.lower() not in raw_answer.lower()

        return QAResponse(
            question=cleaned_question,
            answer=raw_answer,
            evidence=evidence_list,
            model_used=model_used,
            is_available=is_available,
        )
