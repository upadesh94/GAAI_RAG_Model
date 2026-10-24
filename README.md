# 💬 DocChat AI — PDF Document Question Answering System (RAG)

An enterprise-ready, modular, and privacy-conscious AI Question Answering system built with **Retrieval-Augmented Generation (RAG)**. Upload **any PDF document** — research papers, technical manuals, financial reports, legal contracts, business proposals, or resumes — and ask natural-language questions to get accurate, context-grounded answers.

Powered by **Google Gemini** models (`google-genai` SDK) and **ChromaDB** vector storage, DocChat AI ensures that answers are **strictly grounded** in the document's content, eliminates hallucinations, and cites page-level source evidence.

---

## 🌟 Key Highlights

- 📄 **Any PDF Document**: Process papers, reports, manuals, invoices, legal documents, and resumes up to 10 MB.
- 🎨 **Modern Light-Theme Chatbot UI**: Polished Streamlit interface with gradient header, animated message bubbles, document status badges, and suggestion chips.
- 💻 **Terminal CLI Chatbot**: Interactive terminal tool (`cli.py`) for querying PDFs directly from your command line with colored output and page citations.
- 🛡️ **Zero Hallucination Guarantee**: Strict grounding system instructions prevent making up facts. If information is not in the document, it explicitly responds:
  > *"Information not available in the document."*
- 📚 **Page-Level Evidence & Citations**: Every answer provides expandable or inline references to the exact pages and text excerpts used.
- ⚡ **Session-Isolated ChromaDB**: Documents are indexed into isolated collections (`doc_<session_id>`) to prevent cross-document contamination.
- 🧪 **Comprehensive Automated Testing**: 20 unit tests with 100% mock isolation, executable in <1 second without consuming API quota.

---

## 1. System Architecture

```text
       +-------------------------------------------------------------+
       |                     User Interfaces                         |
       |     Streamlit Web Chatbot (app.py)  /  Terminal CLI (cli.py)|
       +------------------------------+------------------------------+
                                      |
                         Upload PDF & Natural-Language Question
                                      |
                                      v
+-------------------------------------+-------------------------------------+
|                             src/doc_qa                                    |
|                                                                           |
|   1. pdf_processing.py                                                    |
|      - Validate format (%PDF header), size limits, encryption             |
|      - Extract page-by-page text with 1-indexed metadata                  |
|      - Chunk text using RecursiveCharacterTextSplitter                    |
|                                                                           |
|   2. embeddings.py (GeminiEmbeddings Adapter)                             |
|      - Google GenAI SDK (gemini-embedding-2 / gemini-embedding-001)        |
|      - In-memory query embedding caching                                  |
|                                                                           |
|   3. vector_store.py (DocVectorStore)                                     |
|      - Isolated ChromaDB collection: doc_<session_id>                     |
|      - Top-k cosine similarity retrieval with relevance scores            |
|      - Safe collection cleanup on session reset                           |
|                                                                           |
|   4. answering.py (DocQAService)                                          |
|      - Strict anti-hallucination grounding prompt                         |
|      - Multi-tier model fallback:                                         |
|        gemini-3.5-flash-lite -> gemini-3.5-flash -> gemini-3.8-flash      |
|      - Returns structured answers + page & excerpt citations              |
+-------------------------------------+-------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                      Google Gemini                          |
       |     - Embeddings: models/gemini-embedding-2                 |
       |     - Generation: models/gemini-3.5-flash-lite / flash      |
       +-------------------------------------------------------------+
```

---

## 2. Technology Stack

| Layer | Component | Details |
|---|---|---|
| **Language** | Python 3.10 – 3.13 | Core language and typed services |
| **LLM & Embeddings** | Google Gemini (`google-genai` SDK) | `gemini-embedding-2` for vectors, `gemini-3.5-flash-lite` for generation |
| **Vector Database** | ChromaDB | Local, session-isolated vector database for semantic search |
| **Chunking Engine** | LangChain Text Splitters | `RecursiveCharacterTextSplitter` preserving page numbers |
| **PDF Extraction** | PyPDF | Zero-dependency PDF parsing, text extraction, and encryption detection |
| **Web Interface** | Streamlit | Modern light-theme chat interface with custom CSS |
| **Terminal CLI** | Python `argparse` + ANSI | Interactive terminal chatbot with rich colored output |
| **Configuration** | `python-dotenv` | API key and environment management via `.env` |
| **Testing** | `pytest` | Automated unit test suite with mock adapters |

---

## 3. Repository Structure

```text
GAAI_RAG_Model/
├── .streamlit/
│   └── config.toml           # Streamlit light theme & server configuration
├── src/
│   ├── app.py                # Streamlit Web Chatbot UI
│   ├── cli.py                # Terminal interactive chatbot
│   └── doc_qa/               # Core modular Python package
│       ├── __init__.py       # Package exports & clean class aliases
│       ├── config.py         # Centralized configuration & safe key masking
│       ├── pdf_processing.py # PDF validation, extraction, & chunking
│       ├── embeddings.py     # GeminiEmbeddings adapter with caching
│       ├── vector_store.py   # Isolated ChromaDB vector store (DocVectorStore)
│       └── answering.py      # Grounded QA service with citations (DocQAService)
├── tests/                    # Automated pytest suite (20 tests)
│   ├── test_pdf_processing.py# PDF extraction, corruption, & chunking tests
│   ├── test_retrieval.py     # Vector retrieval & session isolation tests
│   └── test_answering.py     # Grounded QA, fallback, & API error tests
├── notebooks/                # Jupyter notebook reference & experiments
│   └── GAAI_PR_10_Resume.ipynb
├── app.py                    # Root convenience launcher for Streamlit UI
├── cli.py                    # Root convenience launcher for Terminal CLI
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Production dependencies
├── README.md                 # Complete system documentation
└── .env.example              # Template for environment variables
```

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### Step 1: Clone the Repository
```bash
git clone https://github.com/upadesh94/GAAI_RAG_Model.git
cd GAAI_RAG_Model
```

### Step 2: Create and Activate Virtual Environment
**macOS / Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` with your Gemini API key:
```ini
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Models
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite

# RAG & Chunking Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4

# Storage & File Limits
CHROMA_PERSIST_DIR=.chromadb
MAX_FILE_SIZE_MB=10
```

> **Note:** You can also enter the API key directly in the web app sidebar at runtime.

---

## 5. Usage

### Option A: Web Application (Streamlit)

Launch the web interface:
```bash
streamlit run app.py
```
*(Or `./venv/bin/streamlit run app.py`)*

Open your browser at **`http://localhost:8501`**.

#### Features:
1. **Upload any PDF**: Drag & drop reports, papers, manuals, contracts, or resumes.
2. **Active Status**: View real-time page count and indexed chunk count.
3. **One-Click Suggestions**: Click quick question chips like *"Summarize this document"* or *"List key points"*.
4. **Chat & Citations**: Ask any question and expand the **Supporting Evidence** dropdown to view source excerpts and page numbers.
5. **Sidebar Settings**: Adjust generation models, chunk sizes, retrieval `k`, or click **Reset All** to wipe data.

---

### Option B: Terminal Interactive CLI (`cli.py`)

Ask questions directly from your terminal:

```bash
python cli.py /path/to/your/document.pdf
```
*(Or `./venv/bin/python cli.py /path/to/your/document.pdf`)*

#### Example Session:
```text
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   💬  DocChat CLI — PDF Document Q&A                         ║
║       Powered by Google Gemini & ChromaDB                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

  ℹ  Loading: annual_report.pdf
  ℹ  Extracting text from PDF…
  ✔  Extracted 14 page(s).
  ℹ  Splitting into overlapping chunks…
  ✔  Created 48 text chunks.
  ℹ  Generating embeddings & indexing into ChromaDB…
  ✔  Indexed into collection doc_cli_annual_report.
  ──────────────────────────────────────────────────────────

  Ready! Ask questions about annual_report.pdf.
  Type quit or exit to stop. Type clear to clear screen.

  You ▸ What were the total revenues in Q3?
  AI ▸
     The total revenue for Q3 was $14.2 million, representing an 18%
     growth compared to the same period in the prior year.

  📚 Sources: p.4, p.5  │  Model: gemini-3.5-flash-lite
```

#### CLI Options:
```bash
python cli.py <pdf_path> [options]

Options:
  --model MODEL         Generation model (default: gemini-3.5-flash-lite)
  --embed-model MODEL   Embedding model (default: gemini-embedding-2)
  --chunks K            Number of chunks to retrieve per question (default: 4)
  --chunk-size N        Chunk size in characters (default: 1000)
  --chunk-overlap N     Chunk overlap in characters (default: 200)
  --api-key KEY         Pass API key directly (overrides .env)
```

---

## 6. Example Use Cases & Sample Questions

| Document Type | Example Questions | Expected Grounded Behavior |
|---|---|---|
| **Research Paper** | *"What methodology was used in experiment 2?"*<br/>*"What are the primary conclusions?"* | Cites methodology section, figures, and conclusion paragraphs with page numbers. |
| **Financial Report** | *"What was the net profit margin?"*<br/>*"What risks are highlighted in section 3?"* | Extracts exact financial figures and risk factors mentioned in text. |
| **Technical Manual** | *"How do I configure the webhook endpoint?"*<br/>*"What are the error codes for authentication failure?"* | Surfaces step-by-step instructions and error code tables. |
| **Legal Contract** | *"What is the termination notice period?"*<br/>*"Which jurisdiction governs this agreement?"* | Locates governing law and notice clauses. |
| **Resume / CV** | *"What technical skills does the candidate have?"*<br/>*"What projects did they work on at Company X?"* | Lists only skills and roles explicitly stated in the resume. |
| **Missing Content** | *"Does this document mention quantum computing?"* | If absent, responds: `"Information not available in the document."` |

---

## 7. Automated Testing

All 20 unit tests use deterministic mocks for external Gemini calls, ensuring fast, offline execution without consuming your API quota.

Run the test suite:
```bash
pytest
```
*(Or `./venv/bin/pytest`)*

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.15, pytest-9.1.1
rootdir: GAAI_RAG_Model
configfile: pytest.ini
testpaths: tests

tests/test_answering.py ......                                           [ 30%]
tests/test_pdf_processing.py .........                                   [ 75%]
tests/test_retrieval.py .....                                            [100%]

======================== 20 passed, 1 warning in 0.55s =========================
```

### Coverage Highlights:
- **`test_pdf_processing.py`**: Validates 0-byte detection, invalid magic headers, >10 MB size limits, single/multi-page extraction, scanned image detection, password encryption, and chunk overlapping.
- **`test_retrieval.py`**: Verifies in-memory ChromaDB vector indexing, top-k similarity retrieval, session isolation (preventing cross-document data leaks), and collection deletion.
- **`test_answering.py`**: Checks prompt formatting, strict grounding, citation generation, fallback behavior on absent data, and API error resilience.

---

## 8. Security & Privacy

- **Data Isolation**: Each document session creates an isolated ChromaDB collection. Collections are completely separated and can be wiped with one click (**Reset All**).
- **API Key Safety**: Keys are masked in both the web UI and CLI logs. Keys are kept in session memory and never committed to source control.
- **Git Protection**: [`.gitignore`](.gitignore) excludes `.env`, `.chromadb/`, `venv/`, and any PDF files (`*.pdf`).
- **Data Transmission**: Uploaded document text is transmitted securely over TLS to Google AI Studio for vector embedding and generation.

---

## 9. License & Contributing

Built for educational and enterprise AI evaluation using Google Gemini, ChromaDB, and Streamlit. Contributions, issues, and pull requests are welcome!

Distributed under the MIT License.
