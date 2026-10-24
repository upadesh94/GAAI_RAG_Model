#!/usr/bin/env python3
"""DocChat CLI — Ask questions about any PDF document from the terminal.

Usage:
    python cli.py <path_to_pdf>
    python cli.py --help

Examples:
    python cli.py report.pdf
    python cli.py ~/Documents/research_paper.pdf
    python cli.py /Users/upadesh/Downloads/resume.pdf --chunks 6 --model gemini-3.5-flash
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

# Ensure src/ and root are in sys.path
_src_dir = Path(__file__).resolve().parent
_root_dir = _src_dir.parent
for _p in (str(_src_dir), str(_root_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env before any other imports that depend on env vars
load_dotenv()

from doc_qa.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RETRIEVAL_K,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_GENERATION_MODELS,
)
from doc_qa.embeddings import GeminiEmbeddings, GeminiEmbeddingsError
from doc_qa.pdf_processing import (
    EmptyPDFError,
    EncryptedPDFError,
    PDFProcessingError,
    PDFValidationError,
    ScannedPDFError,
    extract_text_from_pdf,
    split_documents_into_chunks,
)
from doc_qa.vector_store import DocVectorStore, VectorStoreError
from doc_qa.answering import DocQAService, QAResponse


# ── ANSI Colors ─────────────────────────────────────────────────────────────

class C:
    """ANSI color codes for terminal styling."""
    BOLD      = "\033[1m"
    DIM       = "\033[2m"
    ITALIC    = "\033[3m"
    UNDERLINE = "\033[4m"
    RESET     = "\033[0m"
    # Foreground
    PURPLE    = "\033[38;5;99m"
    BLUE      = "\033[38;5;75m"
    CYAN      = "\033[38;5;80m"
    GREEN     = "\033[38;5;114m"
    YELLOW    = "\033[38;5;221m"
    RED       = "\033[38;5;203m"
    GRAY      = "\033[38;5;245m"
    WHITE     = "\033[97m"
    # Background
    BG_PURPLE = "\033[48;5;99m"
    BG_BLUE   = "\033[48;5;24m"


def banner() -> None:
    """Print the startup banner."""
    print(f"""
{C.PURPLE}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   💬  DocChat CLI — PDF Document Q&A                         ║
║       Powered by Google Gemini & ChromaDB                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


def info(msg: str) -> None:
    print(f"  {C.BLUE}ℹ{C.RESET}  {msg}")


def success(msg: str) -> None:
    print(f"  {C.GREEN}✔{C.RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}⚠{C.RESET}  {msg}")


def error(msg: str) -> None:
    print(f"  {C.RED}✖{C.RESET}  {msg}")


def divider() -> None:
    print(f"  {C.GRAY}{'─' * 58}{C.RESET}")


def wrap_text(text: str, width: int = 72, indent: str = "     ") -> str:
    """Word-wrap text for terminal display."""
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if line.strip():
            wrapped.extend(textwrap.wrap(line, width=width, initial_indent=indent, subsequent_indent=indent))
        else:
            wrapped.append("")
    return "\n".join(wrapped)


# ── Core Logic ──────────────────────────────────────────────────────────────

def load_and_index_pdf(
    pdf_path: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> tuple[DocVectorStore, int, int]:
    """Load a PDF, extract text, chunk it, and index into ChromaDB.

    Returns:
        Tuple of (vector_store, page_count, chunk_count).
    """
    path = Path(pdf_path).resolve()
    if not path.is_file():
        error(f"File not found: {path}")
        sys.exit(1)

    filename = path.name
    info(f"Loading: {C.BOLD}{filename}{C.RESET}")

    # 1. Extract text
    info("Extracting text from PDF…")
    pages = extract_text_from_pdf(
        file_input=str(path),
        filename=filename,
    )
    page_count = len(pages)
    success(f"Extracted {C.BOLD}{page_count}{C.RESET} page(s).")

    # 2. Chunk
    info("Splitting into overlapping chunks…")
    chunks = split_documents_into_chunks(
        documents=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunk_count = len(chunks)
    success(f"Created {C.BOLD}{chunk_count}{C.RESET} text chunks.")

    # 3. Embed & index
    info("Generating embeddings & indexing into ChromaDB…")
    embedder = GeminiEmbeddings(api_key=api_key, model=embedding_model)
    vector_store = DocVectorStore(
        embedding_model=embedder,
        persist_directory=".chromadb",
        session_id=f"cli_{path.stem}",
    )
    vector_store.index_documents(chunks)
    success(f"Indexed into collection {C.DIM}{vector_store.collection_name}{C.RESET}.")

    return vector_store, page_count, chunk_count


def ask_loop(
    qa_service: DocQAService,
    retrieval_k: int,
    doc_name: str,
) -> None:
    """Interactive question-answer loop in the terminal."""
    divider()
    print(f"\n  {C.GREEN}{C.BOLD}Ready!{C.RESET} Ask questions about {C.BOLD}{doc_name}{C.RESET}.")
    print(f"  {C.DIM}Type {C.BOLD}quit{C.RESET}{C.DIM} or {C.BOLD}exit{C.RESET}{C.DIM} to stop. "
          f"Type {C.BOLD}clear{C.RESET}{C.DIM} to clear screen.{C.RESET}\n")

    while True:
        try:
            question = input(f"  {C.PURPLE}{C.BOLD}You ▸{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {C.GRAY}Goodbye! 👋{C.RESET}\n")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print(f"\n  {C.GRAY}Goodbye! 👋{C.RESET}\n")
            break
        if question.lower() == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue

        # Get answer
        print(f"\n  {C.CYAN}🤖 Thinking…{C.RESET}", end="", flush=True)
        try:
            response: QAResponse = qa_service.answer_question(
                question=question,
                k=retrieval_k,
            )
            # Clear the "Thinking…" line
            print(f"\r  {' ' * 30}\r", end="")

            # Print answer
            print(f"  {C.CYAN}{C.BOLD}AI ▸{C.RESET}")
            print(wrap_text(response.answer))
            print()

            # Print evidence summary
            if response.evidence:
                print(f"  {C.DIM}📚 Sources: ", end="")
                page_refs = []
                for ev in response.evidence:
                    page_refs.append(f"p.{ev.page}")
                print(", ".join(page_refs), end="")
                print(f"  │  Model: {response.model_used}{C.RESET}")
            elif response.model_used:
                print(f"  {C.DIM}Model: {response.model_used}{C.RESET}")

            print()

        except Exception as e:
            print(f"\r", end="")
            error(f"Failed to answer: {e}\n")


# ── CLI Entry Point ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="docchat",
        description="DocChat CLI — Ask questions about any PDF from the terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python cli.py report.pdf
              python cli.py ~/Documents/paper.pdf --chunks 6
              python cli.py manual.pdf --model gemini-3.5-flash
        """),
    )
    parser.add_argument(
        "pdf",
        type=str,
        help="Path to the PDF document to load.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key. Falls back to GEMINI_API_KEY env var or .env file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=SUPPORTED_GENERATION_MODELS[0],
        choices=SUPPORTED_GENERATION_MODELS,
        help=f"Generation model (default: {SUPPORTED_GENERATION_MODELS[0]}).",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=SUPPORTED_EMBEDDING_MODELS[0],
        choices=SUPPORTED_EMBEDDING_MODELS,
        help=f"Embedding model (default: {SUPPORTED_EMBEDDING_MODELS[0]}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=DEFAULT_RETRIEVAL_K,
        dest="retrieval_k",
        help=f"Number of chunks to retrieve per question (default: {DEFAULT_RETRIEVAL_K}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banner()

    # Resolve API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        error("No Gemini API key found.")
        print(f"  {C.DIM}Set GEMINI_API_KEY in your .env file, environment, or pass --api-key.{C.RESET}\n")
        sys.exit(1)

    info(f"API Key: {C.DIM}{api_key[:4]}…{api_key[-4:]}{C.RESET}")
    info(f"Model:   {C.BOLD}{args.model}{C.RESET}")
    info(f"Embed:   {C.BOLD}{args.embed_model}{C.RESET}")
    info(f"Chunks:  size={args.chunk_size}, overlap={args.chunk_overlap}, k={args.retrieval_k}")
    divider()

    # Load and index the PDF
    try:
        vector_store, page_count, chunk_count = load_and_index_pdf(
            pdf_path=args.pdf,
            api_key=api_key,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embed_model,
        )
    except (PDFProcessingError, GeminiEmbeddingsError, VectorStoreError) as e:
        error(str(e))
        sys.exit(1)

    # Build QA service
    qa_service = DocQAService(
        vector_store=vector_store,
        api_key=api_key,
        model_name=args.model,
    )

    doc_name = Path(args.pdf).name

    # Start interactive loop
    ask_loop(qa_service, args.retrieval_k, doc_name)

    # Cleanup
    try:
        vector_store.clear_data()
        info(f"{C.DIM}Vector store cleaned up.{C.RESET}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
