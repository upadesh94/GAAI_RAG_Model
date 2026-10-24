"""Unit tests for resume_qa.pdf_processing module."""

import io
import pytest
import pypdf
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from resume_qa.pdf_processing import (
    EmptyPDFError,
    EncryptedPDFError,
    PDFValidationError,
    ScannedPDFError,
    extract_text_from_pdf,
    split_documents_into_chunks,
    validate_pdf_bytes,
)


def create_mock_pdf_bytes(
    pages_text: list[str],
    encrypted: bool = False,
    password: str = "secret",
) -> bytes:
    """Helper to generate valid PDF bytes containing specified text on each page."""
    writer = pypdf.PdfWriter()

    for text in pages_text:
        page = writer.add_blank_page(width=400, height=400)
        if text:
            stream = DecodedStreamObject()
            # PDF text stream syntax
            stream.set_data(f"BT /F1 12 Tf 50 350 Td ({text}) Tj ET".encode("latin1"))
            page[NameObject("/Contents")] = stream

            font_dict = DictionaryObject()
            font_dict[NameObject("/Type")] = NameObject("/Font")
            font_dict[NameObject("/Subtype")] = NameObject("/Type1")
            font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")

            res_dict = DictionaryObject()
            fonts = DictionaryObject()
            fonts[NameObject("/F1")] = font_dict
            res_dict[NameObject("/Font")] = fonts
            page[NameObject("/Resources")] = res_dict

    if encrypted:
        writer.encrypt(password)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestPDFValidation:
    """Tests for file byte and format validation."""

    def test_empty_bytes_raises_empty_pdf_error(self):
        with pytest.raises(EmptyPDFError, match="is empty"):
            validate_pdf_bytes(b"", "empty.pdf")

    def test_invalid_header_raises_validation_error(self):
        with pytest.raises(PDFValidationError, match="missing %PDF header"):
            validate_pdf_bytes(b"NOT A REAL PDF FILE", "fake.pdf")

    def test_size_exceeded_raises_validation_error(self):
        large_bytes = b"%PDF-1.4 " + b"0" * (11 * 1024 * 1024)
        with pytest.raises(PDFValidationError, match="exceeds maximum allowed limit"):
            validate_pdf_bytes(large_bytes, "large.pdf", max_size_mb=10)


class TestPDFTextExtraction:
    """Tests for text extraction from valid and invalid PDF files."""

    def test_successful_single_page_extraction(self):
        sample_text = "Alice Smith Senior Software Engineer Skills Python Machine Learning Docker"
        pdf_bytes = create_mock_pdf_bytes([sample_text])

        docs = extract_text_from_pdf(pdf_bytes, filename="alice_resume.pdf")

        assert len(docs) == 1
        assert "Alice Smith" in docs[0].page_content
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["total_pages"] == 1
        assert docs[0].metadata["source"] == "alice_resume.pdf"

    def test_successful_multi_page_extraction(self):
        page1 = "Bob Jones Page One Education BS in Computer Science MIT GPA 3.9"
        page2 = "Bob Jones Page Two Work Experience Staff Engineer at TechCorp 2020-2024"
        pdf_bytes = create_mock_pdf_bytes([page1, page2])

        docs = extract_text_from_pdf(pdf_bytes, filename="bob_resume.pdf")

        assert len(docs) == 2
        assert docs[0].metadata["page"] == 1
        assert "Page One" in docs[0].page_content
        assert docs[1].metadata["page"] == 2
        assert "Page Two" in docs[1].page_content
        assert docs[0].metadata["total_pages"] == 2
        assert docs[1].metadata["total_pages"] == 2

    def test_scanned_pdf_without_text_raises_scanned_pdf_error(self):
        # Create PDF with blank pages (no extractable text)
        pdf_bytes = create_mock_pdf_bytes(["", ""])

        with pytest.raises(ScannedPDFError, match="scanned or image-only"):
            extract_text_from_pdf(pdf_bytes, filename="scanned.pdf")

    def test_encrypted_pdf_raises_encrypted_pdf_error(self):
        sample_text = "Confidential resume information for encrypted candidate"
        pdf_bytes = create_mock_pdf_bytes([sample_text], encrypted=True, password="lock")

        with pytest.raises(EncryptedPDFError, match="password-protected|encrypted"):
            extract_text_from_pdf(pdf_bytes, filename="encrypted.pdf")


class TestTextChunking:
    """Tests for RecursiveCharacterTextSplitter and metadata tracking."""

    def test_split_documents_preserves_metadata(self):
        page1 = "Candidate Skill Summary: " + ("Python FastAPI PostgreSQL Docker " * 40)
        page2 = "Work History: " + ("Developed scalable microservices with Kubernetes " * 40)
        pdf_bytes = create_mock_pdf_bytes([page1, page2])

        pages = extract_text_from_pdf(pdf_bytes, filename="skills.pdf")
        chunks = split_documents_into_chunks(pages, chunk_size=300, chunk_overlap=50)

        assert len(chunks) > 2
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_id"] == idx + 1
            assert chunk.metadata["source"] == "skills.pdf"
            assert chunk.metadata["page"] in (1, 2)
            assert len(chunk.page_content) > 0

    def test_empty_documents_returns_empty_list(self):
        assert split_documents_into_chunks([]) == []
