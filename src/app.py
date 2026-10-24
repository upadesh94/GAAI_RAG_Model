"""DocChat AI — PDF Document Question Answering System (Streamlit App).

A modern, professional chatbot interface that allows users to upload any PDF
document, index its content into ChromaDB, and ask natural language questions
grounded in the document text via Google Gemini.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

# Ensure src/ and root are in sys.path
_src_dir = Path(__file__).resolve().parent
_root_dir = _src_dir.parent
for _p in (str(_src_dir), str(_root_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from doc_qa.config import (
    AppConfig,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_RETRIEVAL_K,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_GENERATION_MODELS,
    get_default_config,
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
from doc_qa.answering import (
    FALLBACK_NOT_AVAILABLE,
    DocQAService,
    EvidenceItem,
    QAResponse,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample prompt suggestions (generic for any document)
SUGGESTED_QUESTIONS = [
    "📋 Summarize this document",
    "🔑 What are the key points?",
    "📊 List any data or statistics",
    "🧑 Who are the main people mentioned?",
    "📅 What dates or timelines are referenced?",
    "⚖️ What conclusions are drawn?",
]

# Page configuration
st.set_page_config(
    page_title="DocChat AI — PDF Q&A",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Professional light-theme chatbot interface
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Main layout adjustments ─── */
    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ── Header area ────────────────────────────────────────── */
    .header-container {
        background: linear-gradient(135deg, #6366F1 0%, #38BDF8 100%);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    }
    .header-container h1 {
        margin: 0 0 6px 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    .header-container p {
        margin: 0;
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.92);
        font-weight: 400;
    }

    /* ── Document status card ───────────────────────────────── */
    .doc-status-card {
        display: flex;
        align-items: center;
        gap: 14px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 18px;
        min-height: 82px;
    }
    .doc-status-card.active {
        background: #EEF2FF;
        border-color: #C7D2FE;
    }
    .doc-status-card .icon-col {
        font-size: 1.8rem;
        line-height: 1;
    }
    .doc-status-card .label-tag {
        font-size: 0.72rem;
        color: #4F46E5;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .doc-status-card .doc-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #1E293B;
        word-break: break-all;
    }
    .doc-status-card .doc-meta {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* ── Chat messages ──────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        margin-bottom: 12px !important;
        padding: 14px 18px !important;
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #EEF2FF !important;
        border-color: #E0E7FF !important;
    }

    /* ── Evidence box ───────────────────────────────────────── */
    .evidence-box {
        border-left: 3px solid #6366F1;
        background: #F8FAFC;
        padding: 10px 14px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.86rem;
        line-height: 1.5;
        color: #334155;
    }
    .evidence-box strong {
        color: #4338CA;
    }

    /* ── Suggestion buttons ─────────────────────────────────── */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px !important;
        font-size: 0.82rem !important;
        padding: 4px 12px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        color: #334155 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #6366F1 !important;
        color: #6366F1 !important;
        background-color: #EEF2FF !important;
    }

    /* ── Empty state ────────────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 50px 20px;
        color: #64748B;
    }
    .empty-state .icon {
        font-size: 3rem;
        margin-bottom: 12px;
    }
    .empty-state h3 {
        color: #1E293B;
        margin: 0 0 6px 0;
        font-weight: 600;
        font-size: 1.25rem;
    }
    .empty-state p {
        font-size: 0.9rem;
        color: #64748B;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Model badge ────────────────────────────────────────── */
    .model-badge {
        display: inline-block;
        font-size: 0.72rem;
        color: #64748B;
        background: #F1F5F9;
        padding: 2px 8px;
        border-radius: 6px;
        margin-top: 6px;
        font-weight: 500;
    }

    /* ── Footer ─────────────────────────────────────────────── */
    .footer-note {
        text-align: center;
        font-size: 0.75rem;
        color: #94A3B8;
        padding: 30px 0 10px;
    }
    .footer-note {
        text-align: center;
        font-size: 0.75rem;
        color: #CBD5E1;
        padding: 20px 0 10px;
        letter-spacing: 0.3px;
    }

    /* hide default streamlit footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]

    if "config" not in st.session_state:
        st.session_state.config = get_default_config()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None

    if "current_file_bytes" not in st.session_state:
        st.session_state.current_file_bytes = None

    if "page_count" not in st.session_state:
        st.session_state.page_count = 0

    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "qa_service" not in st.session_state:
        st.session_state.qa_service = None

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def clear_document_and_store() -> None:
    """Clear the active document, delete its isolated vector store, and reset state."""
    if st.session_state.vector_store is not None:
        try:
            st.session_state.vector_store.clear_data()
        except Exception as e:
            logger.warning("Error clearing vector store: %s", e)

    st.session_state.vector_store = None
    st.session_state.qa_service = None
    st.session_state.current_file_name = None
    st.session_state.current_file_bytes = None
    st.session_state.page_count = 0
    st.session_state.chunk_count = 0
    st.session_state.messages = []
    st.session_state.pending_question = None
    # Generate new session ID for complete isolation
    st.session_state.session_id = uuid.uuid4().hex[:12]


def render_sidebar() -> None:
    """Render sidebar configuration, status, and controls."""
    with st.sidebar:
        st.markdown("## 🛠️ Settings")

        # API Key input / detection
        env_key = os.getenv("GEMINI_API_KEY", "")
        has_env_key = bool(env_key and env_key.strip())

        user_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.config.api_key or "",
            placeholder="Paste your Gemini API key…",
            help="Your key is kept only in session memory and never saved to public storage.",
        )

        if user_key and user_key.strip():
            st.session_state.config.api_key = user_key.strip()
            st.caption(f"🔒 Active: `{st.session_state.config.get_masked_api_key()}`")
        elif has_env_key:
            st.session_state.config.api_key = env_key.strip()
            st.caption(f"🔒 From `.env`: `{st.session_state.config.get_masked_api_key()}`")
        else:
            st.warning("⚠️ Add a Gemini API key to start chatting.")

        st.divider()

        # Model configuration
        st.markdown("#### 🤖 Models")
        selected_gen_model = st.selectbox(
            "Generation Model",
            options=SUPPORTED_GENERATION_MODELS,
            index=0,
            help="Gemini model used to generate grounded answers.",
        )
        st.session_state.config.generation_model = selected_gen_model

        selected_embed_model = st.selectbox(
            "Embedding Model",
            options=SUPPORTED_EMBEDDING_MODELS,
            index=0,
            help="Gemini model used for vector representations.",
        )
        st.session_state.config.embedding_model = selected_embed_model

        st.divider()

        # RAG Parameters
        with st.expander("⚙️ RAG Parameters", expanded=False):
            st.session_state.config.chunk_size = st.slider(
                "Chunk Size (chars)",
                min_value=300,
                max_value=2500,
                value=st.session_state.config.chunk_size,
                step=100,
            )
            st.session_state.config.chunk_overlap = st.slider(
                "Chunk Overlap (chars)",
                min_value=50,
                max_value=500,
                value=st.session_state.config.chunk_overlap,
                step=25,
            )
            st.session_state.config.retrieval_k = st.slider(
                "Retrieved Chunks (k)",
                min_value=1,
                max_value=10,
                value=st.session_state.config.retrieval_k,
                step=1,
            )

        st.divider()

        # Session & Data Management
        st.markdown("#### 🗂️ Session")
        st.caption(f"ID: `{st.session_state.session_id}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🗑️ Reset All", use_container_width=True):
                clear_document_and_store()
                st.success("All data cleared.")
                st.rerun()

        # Privacy notice
        with st.expander("🛡️ Privacy & Disclaimers"):
            st.markdown(
                """
                - **Data Processing**: Uploaded document text is sent to Google's Gemini API for embedding and generation.
                - **Confidentiality**: Do not upload unconsented or strictly confidential personal data.
                - **Accuracy**: Answers are for informational assistance only; always verify critical information.
                - **Storage**: Chunks are stored in an isolated Chroma collection tied to this active session.
                """
            )


def process_uploaded_document(uploaded_file) -> None:
    """Validate, extract, chunk, and index the uploaded PDF document."""
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    # Check if this exact file is already indexed
    if (
        st.session_state.current_file_name == filename
        and st.session_state.current_file_bytes == file_bytes
        and st.session_state.vector_store is not None
    ):
        return

    # Check API key before processing
    if not st.session_state.config.api_key:
        st.error("❌ Please provide a valid Gemini API key in the sidebar before uploading.")
        return

    with st.status(f"Processing '{filename}'…", expanded=True) as status:
        try:
            # 1. Validation and Text Extraction
            status.write("🔍 Validating PDF format and integrity…")
            pages = extract_text_from_pdf(
                file_input=file_bytes,
                filename=filename,
                max_size_mb=st.session_state.config.max_file_size_mb,
            )
            page_count = len(pages)
            status.write(f"📄 Extracted {page_count} page(s).")

            # 2. Text Chunking
            status.write("✂️ Splitting text into overlapping chunks…")
            chunks = split_documents_into_chunks(
                documents=pages,
                chunk_size=st.session_state.config.chunk_size,
                chunk_overlap=st.session_state.config.chunk_overlap,
            )
            chunk_count = len(chunks)
            status.write(f"🧩 Created {chunk_count} text chunks.")

            # 3. Vector Embeddings & Indexing
            status.write("⚡ Generating embeddings and indexing into ChromaDB…")
            embedding_model = GeminiEmbeddings(
                api_key=st.session_state.config.api_key,
                model=st.session_state.config.embedding_model,
            )

            # Create isolated vector store for this session
            vector_store = DocVectorStore(
                embedding_model=embedding_model,
                persist_directory=st.session_state.config.chroma_persist_dir,
                session_id=st.session_state.session_id,
            )
            vector_store.index_documents(chunks)

            # 4. Initialize QA Service
            qa_service = DocQAService(
                vector_store=vector_store,
                api_key=st.session_state.config.api_key,
                model_name=st.session_state.config.generation_model,
            )

            # Update session state
            st.session_state.vector_store = vector_store
            st.session_state.qa_service = qa_service
            st.session_state.current_file_name = filename
            st.session_state.current_file_bytes = file_bytes
            st.session_state.page_count = page_count
            st.session_state.chunk_count = chunk_count

            status.update(
                label=f"✅ '{filename}' indexed — {page_count} pages, {chunk_count} chunks",
                state="complete",
                expanded=False,
            )

        except EmptyPDFError as e:
            status.update(label="❌ Empty PDF", state="error")
            st.error(f"Validation Error: {e}")
        except EncryptedPDFError as e:
            status.update(label="❌ Encrypted PDF", state="error")
            st.error(f"Encryption Error: {e}")
        except ScannedPDFError as e:
            status.update(label="❌ Scanned / Image PDF", state="error")
            st.error(f"Scanned Document: {e}")
        except PDFValidationError as e:
            status.update(label="❌ PDF validation failed", state="error")
            st.error(f"Invalid PDF: {e}")
        except GeminiEmbeddingsError as e:
            status.update(label="❌ Embedding error", state="error")
            st.error(f"Gemini API Error: {e}")
        except Exception as e:
            status.update(label="❌ Processing failed", state="error")
            logger.exception("Unexpected error during document processing")
            st.error(f"Processing failed: {type(e).__name__} — {e}")


def handle_question(question_text: str) -> None:
    """Execute question answering and append to chat history."""
    if not question_text or not question_text.strip():
        return

    qa_service = st.session_state.qa_service
    if not qa_service:
        st.warning("Please upload a PDF document first before asking questions.")
        return

    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": question_text})

    # Generate answer
    with st.spinner("Thinking…"):
        try:
            response: QAResponse = qa_service.answer_question(
                question=question_text,
                k=st.session_state.config.retrieval_k,
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "evidence": [
                        {
                            "page": item.page,
                            "source": item.source,
                            "excerpt": item.excerpt,
                            "score": item.score,
                        }
                        for item in response.evidence
                    ],
                    "model_used": response.model_used,
                    "is_available": response.is_available,
                }
            )
        except Exception as e:
            logger.exception("Failed to answer question")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"An error occurred while answering: {e}",
                    "evidence": [],
                    "model_used": "",
                    "is_available": False,
                }
            )


# ── Main App ────────────────────────────────────────────────────────────────

def main() -> None:
    """Main Streamlit application entry point."""
    init_session_state()
    render_sidebar()

    # ── Header ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="header-container">
            <h1>💬 DocChat AI</h1>
            <p>Upload any PDF document and ask questions — powered by RAG with Google Gemini & ChromaDB</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Document upload & status row ────────────────────────
    upload_col, status_col = st.columns([3, 2])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a PDF Document",
            type=["pdf"],
            help="Upload any text-based PDF (max 10 MB) — reports, papers, manuals, resumes, and more.",
        )

    with status_col:
        if st.session_state.current_file_name:
            st.markdown(
                f"""
                <div class="doc-status-card active">
                    <div class="icon-col">📄</div>
                    <div>
                        <div class="label-tag">Active Document</div>
                        <div class="doc-title">{st.session_state.current_file_name}</div>
                        <div class="doc-meta">{st.session_state.page_count} pages · {st.session_state.chunk_count} chunks indexed</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="doc-status-card">
                    <div class="icon-col">📂</div>
                    <div>
                        <div class="label-tag" style="color:#94A3B8;">No Document</div>
                        <div class="doc-title" style="color:#64748B; font-weight:400;">Upload a PDF to get started</div>
                        <div class="doc-meta">Supports any text PDF up to 10 MB</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Process upload if provided
    if uploaded_file is not None:
        process_uploaded_document(uploaded_file)

    st.divider()

    # ── Suggestion chips (shown when document is active) ────
    if st.session_state.current_file_name and st.session_state.qa_service:
        st.markdown("**💡 Try asking:**")
        cols = st.columns(len(SUGGESTED_QUESTIONS[:3]))
        for idx, q in enumerate(SUGGESTED_QUESTIONS[:3]):
            if cols[idx].button(q, key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        cols2 = st.columns(len(SUGGESTED_QUESTIONS[3:]))
        for idx, q in enumerate(SUGGESTED_QUESTIONS[3:]):
            if cols2[idx].button(q, key=f"sug2_{idx}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        st.markdown("")  # spacing

    # ── Chat history ────────────────────────────────────────
    if not st.session_state.messages and not st.session_state.current_file_name:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">💬</div>
                <h3>Welcome to DocChat AI</h3>
                <p>Upload any PDF document and start asking questions.<br/>
                Your data stays private and is processed in an isolated session.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not st.session_state.messages and st.session_state.current_file_name:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">✨</div>
                <h3>Document ready!</h3>
                <p>Try one of the suggested questions above, or type your own below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
            st.markdown(msg["content"])

            # Evidence citations
            evidence = msg.get("evidence", [])
            if evidence:
                with st.expander(f"📚 {len(evidence)} source excerpts"):
                    for i, ev in enumerate(evidence):
                        st.markdown(
                            f'<div class="evidence-box">'
                            f'<strong>Excerpt {i+1} — Page {ev.get("page", "?")}</strong>'
                            f'<br><span style="color:#94A3B8; font-size:0.78rem;">Source: {ev.get("source", "document.pdf")}</span>'
                            f'<p style="margin-top:6px; white-space:pre-wrap;">{ev.get("excerpt", "")}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            # Model badge
            model_used = msg.get("model_used")
            if model_used:
                st.markdown(
                    f'<span class="model-badge">🤖 {model_used}</span>',
                    unsafe_allow_html=True,
                )

    # Handle pending suggested question if clicked
    if st.session_state.pending_question:
        q_to_run = st.session_state.pending_question
        st.session_state.pending_question = None
        handle_question(q_to_run)
        st.rerun()

    # ── Chat input ──────────────────────────────────────────
    prompt = st.chat_input("Ask anything about your document…")
    if prompt:
        handle_question(prompt)
        st.rerun()

    # ── Footer ──────────────────────────────────────────────
    st.markdown(
        '<div class="footer-note">DocChat AI · RAG powered by Google Gemini & ChromaDB</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
