"""PDF text extraction, validation, and chunking module for resumes.

Handles file validation (size, encryption, corruption), page-by-page text extraction
with metadata preservation, detection of scanned/image-only PDFs, and
configurable chunking using RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Optional, Union
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessingError(Exception):
    """Base exception for PDF processing failures."""


class PDFValidationError(PDFProcessingError):
    """Raised when PDF file format or size validation fails."""


class EncryptedPDFError(PDFProcessingError):
    """Raised when the PDF file is encrypted or password-protected."""


class ScannedPDFError(PDFProcessingError):
    """Raised when the PDF contains pages but no extractable text (scanned/image-only)."""


class EmptyPDFError(PDFProcessingError):
    """Raised when the PDF has 0 bytes or 0 pages."""


def validate_pdf_bytes(
    file_bytes: bytes,
    filename: str = "resume.pdf",
    max_size_mb: int = 10,
) -> None:
    """Validate raw PDF bytes before processing.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Name of the file for reporting.
        max_size_mb: Maximum permitted size in megabytes.

    Raises:
        PDFValidationError: If format or size is invalid.
        EmptyPDFError: If the file is 0 bytes.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise EmptyPDFError(f"Uploaded file '{filename}' is empty (0 bytes).")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise PDFValidationError(
            f"File '{filename}' size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB."
        )

    # Validate PDF signature/magic bytes
    if not file_bytes.startswith(b"%PDF-"):
        raise PDFValidationError(
            f"File '{filename}' does not appear to be a valid PDF (missing %PDF header)."
        )


def extract_text_from_pdf(
    file_input: Union[bytes, io.BytesIO, str, Path],
    filename: str = "resume.pdf",
    max_size_mb: int = 10,
) -> List[Document]:
    """Extract page-by-page text from a PDF with metadata preservation.

    Args:
        file_input: File path, bytes, or BytesIO buffer of the PDF.
        filename: Label for metadata tracking.
        max_size_mb: File size limit.

    Returns:
        List of Document objects, one per non-empty page, with page metadata (1-indexed).

    Raises:
        PDFValidationError: If file is not a valid PDF or exceeds size limit.
        EncryptedPDFError: If file is encrypted with a password.
        ScannedPDFError: If no extractable text is present (scanned/image-only).
        EmptyPDFError: If file has no pages or is empty.
    """
    raw_bytes: bytes
    if isinstance(file_input, (str, Path)):
        path = Path(file_input)
        if not path.is_file():
            raise PDFValidationError(f"File not found: {path}")
        filename = path.name
        raw_bytes = path.read_bytes()
    elif isinstance(file_input, io.BytesIO):
        raw_bytes = file_input.getvalue()
    elif isinstance(file_input, bytes):
        raw_bytes = file_input
    else:
        raise TypeError(f"Unsupported file_input type: {type(file_input)}")

    # Run validation
    validate_pdf_bytes(raw_bytes, filename=filename, max_size_mb=max_size_mb)

    stream = io.BytesIO(raw_bytes)
    try:
        reader = pypdf.PdfReader(stream)
    except Exception as e:
        raise PDFValidationError(f"Could not read PDF '{filename}': {e}") from e

    if reader.is_encrypted:
        try:
            # Attempt empty-password decrypt
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise EncryptedPDFError(
                    f"PDF '{filename}' is password-protected. Please upload an unencrypted file."
                )
        except Exception as e:
            raise EncryptedPDFError(
                f"PDF '{filename}' is encrypted and cannot be opened without a password."
            ) from e

    num_pages = len(reader.pages)
    if num_pages == 0:
        raise EmptyPDFError(f"PDF '{filename}' contains 0 pages.")

    pages_docs: List[Document] = []
    total_text_length = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = page.extract_text() or ""
        cleaned_text = page_text.strip()
        total_text_length += len(cleaned_text)

        pages_docs.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source": filename,
                    "page": page_num,
                    "total_pages": num_pages,
                    "char_count": len(cleaned_text),
                },
            )
        )

    # If the combined text across all pages is extremely short or empty, it's likely scanned
    if total_text_length < 15:
        raise ScannedPDFError(
            f"No readable text could be extracted from '{filename}'. "
            "The PDF may be scanned or image-only, which requires Optical Character Recognition (OCR)."
        )

    return pages_docs


def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split extracted page documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Preserves source and page metadata for evidence tracking.

    Args:
        documents: List of Document objects from extract_text_from_pdf.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap size between adjacent chunks.

    Returns:
        List of Document chunks with chunk_id, page, and source in metadata.
    """
    if not documents:
        return []

    # Filter out empty pages before splitting
    non_empty_docs = [doc for doc in documents if doc.page_content.strip()]
    if not non_empty_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_documents(non_empty_docs)

    chunks: List[Document] = []
    for idx, chunk in enumerate(raw_chunks):
        cleaned_content = chunk.page_content.strip()
        if not cleaned_content:
            continue

        meta = dict(chunk.metadata)
        meta["chunk_id"] = idx + 1
        chunks.append(
            Document(
                page_content=cleaned_content,
                metadata=meta,
            )
        )

    return chunks
