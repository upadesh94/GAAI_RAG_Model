# AI Resume Question Answering System Using RAG

An enterprise-ready, modular, and privacy-conscious AI Resume Question Answering System built with Retrieval-Augmented Generation (RAG). Upload PDF resumes and ask natural-language questions regarding a candidate's education, skills, work experience, certifications, and projects.

Powered by **Google Gemini** models via the modern `google-genai` SDK and **ChromaDB** vector storage, the system ensures that responses are **strictly grounded** in the resume text, preventing hallucination while citing page-level evidence.

---

## 1. Overview & Problem Statement

### The Problem
Reviewing resumes manually to verify qualifications, identify key technical skills, evaluate project depths, or check tenure across multiple candidates is tedious and time-consuming. Traditional keyword searches frequently fail to capture contextual semantics (e.g., distinguishing between tools used in production vs. academic coursework).

### The Solution
The **AI Resume Question Answering System** applies Retrieval-Augmented Generation (RAG) to convert unstructured PDF resumes into searchable vector embeddings. When a recruiter, hiring manager, or interviewer asks a question, the system retrieves only the most relevant resume excerpts and prompts Google Gemini to formulate an accurate, concise answer backed by exact page citations. If information is not found in the resume, the system explicitly reports:
> `"Information not available in the resume."`

---

## 2. Objectives & Features

### Core Implemented Features
- **PDF Upload & Deep Validation**:
  - Validates file type (%PDF magic header), corruption, and file size limits (default 10 MB).
  - Detects encrypted and password-protected files gracefully.
  - Automatically identifies scanned or image-only PDFs lacking extractable text, notifying the user.
- **Page-Aware Text Extraction & Metadata Tracking**:
  - Extracts text page-by-page using `pypdf`.
  - Preserves 1-indexed page numbers, source filename, and character counts.
- **Configurable Text Chunking**:
  - Splits text into overlapping segments using `RecursiveCharacterTextSplitter`.
  - Configurable chunk size (default: 1000) and chunk overlap (default: 200).
- **Gemini Embeddings Adapter with In-Memory Caching**:
  - Reusable adapter implementing LangChain's embedding interface using `google-genai` SDK.
  - In-memory cache eliminates duplicate embedding calls for identical queries.
  - Never exposes API keys in logs or user interfaces.
- **Session-Isolated ChromaDB Vector Storage**:
  - Each uploaded resume is indexed into an isolated collection identified by a unique session ID (`resume_<session_id>`).
  - Completely avoids cross-resume contamination between candidates or browser tabs.
  - Provides a one-click **Reset All** action to safely delete ChromaDB collections and clear session memory.
- **Context-Grounded Question Answering**:
  - Strict system instruction prevents hallucination and answers from general knowledge.
  - Automatic model fallback across flash tiers (`gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.8-flash`) if an individual model encounters temporary demand spikes.
  - Returns structured responses with citations (page numbers, similarity ranks, and text excerpts).
- **Interactive Streamlit Web Application**:
  - Clean, modern UI with responsive chat layout.
  - Interactive suggested question chips for quick testing.
  - Expandable evidence section revealing exact excerpts and source pages.
  - Sidebar controls for API key management, model selection, and RAG tuning.
- **Comprehensive Automated Test Suite**:
  - 20 automated tests using `pytest` verifying extraction, chunking, retrieval, isolation, and error handling.
  - Fully mocked unit tests for offline execution without consuming API quota.

---

## 3. Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.10+ | Core language for backend logic and services |
| **LLM & Embeddings** | Google Gemini (`google-genai` SDK) | `gemini-embedding-2` for embeddings, `gemini-3.5-flash-lite` / `gemini-3.5-flash` for generation |
| **Vector Database** | ChromaDB | Local, session-isolated vector database for chunk storage and cosine similarity search |
| **Chunking & RAG Core** | LangChain Text Splitters & Community | `RecursiveCharacterTextSplitter` and Chroma integrations |
| **PDF Extraction** | PyPDF | Fast, zero-dependency PDF parsing, text extraction, and encryption detection |
| **Frontend UI** | Streamlit | Responsive web application with chat interface and sidebar controls |
| **Configuration** | `python-dotenv` | Secure API key and parameter management via `.env` |
| **Testing** | `pytest` | Automated unit and integration testing suite |

---

## 4. System Architecture

```text
       +-------------------------------------------------------------+
       |                     User Web Interface                      |
       |                (Streamlit Responsive UI)                    |
       +------------------------------+------------------------------+
                                      |
                     Upload PDF Resume / User Question
                                      |
                                      v
+-------------------------------------+-------------------------------------+
|                              resume_qa                                     |
|                                                                           |
|   1. pdf_processing.py                                                    |
|      - Validate header, size, encryption, scanned PDFs                    |
|      - Extract text per page with 1-indexed metadata                      |
|      - Chunk text (RecursiveCharacterTextSplitter)                        |
|                                                                           |
|   2. embeddings.py (GeminiEmbeddings Adapter)                             |
|      - google-genai SDK (gemini-embedding-2)                              |
|      - Query embedding caching                                            |
|                                                                           |
|   3. vector_store.py (ResumeVectorStore)                                  |
|      - Isolated ChromaDB collection: resume_<session_id>                  |
|      - Top-k similarity retrieval with scores and metadata                |
|      - Collection deletion on session reset                               |
|                                                                           |
|   4. answering.py (ResumeQAService)                                       |
|      - Strict grounding system instruction                                |
|      - Context formatting with page citations                             |
|      - Fallback: "Information not available in the resume."               |
|      - Multi-model fallback resilient to 503 spikes                       |
+-------------------------------------+-------------------------------------+
                                      |
                   API Requests (HTTPS with masked key)
                                      v
       +-------------------------------------------------------------+
       |                      Google Gemini API                      |
       |     - Embeddings: models/gemini-embedding-2                 |
       |     - Generation: models/gemini-3.5-flash-lite / flash      |
       +-------------------------------------------------------------+
```

---

## 5. Repository Structure

```text
ai-resume-qa/
├── .env.example              # Template for environment variables
├── .gitignore                # Git exclusions (API keys, caches, ChromaDB, PDFs)
├── README.md                 # Complete system documentation
├── requirements.txt          # Python dependencies
├── app.py                    # Streamlit web application
├── resume_qa/                # Modular Python package
│   ├── __init__.py           # Package exports
│   ├── config.py             # Config & settings with validation and masking
│   ├── pdf_processing.py     # PDF validation, extraction, and chunking
│   ├── embeddings.py         # GeminiEmbeddings adapter with caching
│   ├── vector_store.py       # Isolated ChromaDB vector store
│   └── answering.py          # Grounded QA service with citations & fallback
├── tests/                    # Automated pytest suite
│   ├── test_pdf_processing.py# PDF extraction, corruption, & chunking tests
│   ├── test_retrieval.py     # Vector retrieval & session isolation tests
│   └── test_answering.py     # Grounded QA, fallback, & API error tests
├── notebooks/                # Functional reference notebook
│   └── GAAI_PR_10_Resume.ipynb
└── data/                     # Local test datasets (ignored by git)
    └── sample_resume.pdf     # Synthetic 2-page test resume
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10 to 3.13 installed.
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### Step 1: Clone or Open Repository
```bash
git clone <repository-url>
cd ai-resume-qa
```

### Step 2: Create and Activate Virtual Environment (Recommended)
**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` to supply your configuration:
```ini
# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Configurable Model Identifiers
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite

# RAG & Chunking Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4

# Storage & File Settings
CHROMA_PERSIST_DIR=.chromadb
MAX_FILE_SIZE_MB=10
```

> **Note:** Alternatively, you can enter the Gemini API key directly in the Streamlit sidebar.

---

## 8. Running the Application

Launch the Streamlit web application:

```bash
python -m streamlit run app.py
```

*(Or `streamlit run app.py` if Python's `Scripts/` folder is in your system PATH).*

The application will launch in your default web browser at `http://localhost:8501`.

### User Workflow
1. Enter or verify your Gemini API key in the sidebar.
2. Drag and drop a PDF resume (e.g., `data/sample_resume.pdf`).
3. Monitor the processing status indicators (page extraction, chunking, embedding).
4. Click any of the **Suggested Questions** or type a custom question in the chat bar.
5. Review the grounded answer along with expandable **Supporting Evidence** excerpts and page numbers.
6. Use **Clear Chat** or **Reset All** to wipe data when finished.

---

## 9. Example Questions & Output

After uploading a candidate resume, try questions such as:

| Question | Expected System Behavior |
|---|---|
| *"What programming languages does the candidate know?"* | Retrieves skills section; lists only languages explicitly written in the resume. |
| *"What is the candidate's educational qualification?"* | Retrieves education section; cites degree, university, and graduation year. |
| *"What technical skills does the candidate have?"* | Surfaces frameworks, databases, cloud platforms, and developer tools. |
| *"What projects has the candidate worked on?"* | Summarizes project titles, descriptions, and measurable outcomes. |
| *"Which certifications are mentioned?"* | Lists AWS, Google Cloud, or other verified certifications. |
| *"Does the candidate have experience flying airplanes?"* | If absent, responds with: `Information not available in the resume.` |

---

## 10. Automated Testing

All tests use mocks for external Gemini API calls, allowing fast, deterministic offline execution without consuming API quota.

Run the entire test suite:
```bash
python -m pytest -v
```

### Test Coverage Highlights
- `tests/test_pdf_processing.py`:
  - `test_empty_bytes_raises_empty_pdf_error`: 0-byte validation.
  - `test_invalid_header_raises_validation_error`: Invalid magic bytes.
  - `test_size_exceeded_raises_validation_error`: Rejection of files > 10 MB.
  - `test_successful_single_page_extraction`: Page 1 content & metadata.
  - `test_successful_multi_page_extraction`: Page 1 & 2 ordering and page counts.
  - `test_scanned_pdf_without_text_raises_scanned_pdf_error`: Scanned image detection.
  - `test_encrypted_pdf_raises_encrypted_pdf_error`: Password protection handling.
  - `test_split_documents_preserves_metadata`: Chunk size, overlap, and metadata.
- `tests/test_retrieval.py`:
  - `test_indexing_and_retrieval`: In-memory ChromaDB search.
  - `test_session_isolation_prevents_cross_resume_contamination`: Confirms Candidate A cannot see Candidate B's chunks.
  - `test_retrieve_without_indexing_raises_error`: Unindexed state check.
  - `test_clear_data_resets_store`: Clean data wiping and reset.
- `tests/test_answering.py`:
  - `test_missing_api_configuration_raises_error`: Configuration check.
  - `test_empty_question_returns_friendly_message`: Clean user guidance.
  - `test_empty_retrieval_returns_fallback_without_calling_gemini`: Quota saving.
  - `test_grounded_answer_with_evidence`: Verification of grounded prompt and citations.
  - `test_missing_information_returns_fallback`: Fallback phrase enforcement.
  - `test_external_api_failure_handled_gracefully`: Graceful error recovery.

---

## 11. Security & Privacy Guidelines

- **Personal Data Handling**: Resumes contain Personally Identifiable Information (PII) such as full names, email addresses, phone numbers, and employment history.
- **Third-Party Processing**: Extracted resume text is transmitted to Google's Gemini API endpoints for embedding and generation. Ensure candidate consent before processing third-party resumes.
- **No Git Commit of Resumes or Keys**: `.gitignore` strictly excludes `.env`, `data/*.pdf`, uploads, and local ChromaDB files (`.chromadb/`).
- **Data Deletion**: Use the **Reset All** button to immediately purge the active session's collection from ChromaDB and clear in-memory state.
- **Hiring Decision Disclaimer**: This software is an informational retrieval tool. Generated responses should **never** serve as the sole criterion for hiring, promotion, or rejection decisions.

---

## 12. Known Limitations & Future Improvements

### Current Limitations
1. **Scanned PDFs / OCR**: The current extraction engine relies on text layers. Scanned or image-only PDFs are detected and rejected with a helpful warning.
2. **Complex Tables & Multi-column Layouts**: Unconventional visual formatting (e.g., dual-column sidebars) may result in out-of-order text reading without specialized visual layout parsing.
3. **External API Dependence**: Embeddings and answer generation require outbound internet connectivity to Google AI Studio.

### Planned Enhancements
- [ ] Integration with `Tesseract` or `easyocr` for automatic OCR on scanned resumes.
- [ ] Multi-resume comparison mode (e.g., comparing candidate A vs candidate B side-by-side).
- [ ] Structured JSON export of candidate profile (education, skills, years of experience).
- [ ] PDF bounding box highlighting showing the exact visual location of evidence in the PDF viewer.
- [ ] Automated evaluation suite measuring retrieval recall and grounding precision.

---

## 13. License & Contributing

Built for educational and enterprise AI evaluation using Google Gemini, ChromaDB, and Streamlit. Contributions and pull requests are welcome!
