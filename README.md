# 🔒 Local Document Q&A (Private RAG Pipeline)

A 100% local, private Retrieval-Augmented Generation (RAG) web application. Upload documents (PDF, DOCX, XLSX, CSV, JSON, TXT, Markdown, Python, JavaScript, or TypeScript), chunk and embed them using local Ollama models, store vectors in an embedded LanceDB database, and ask natural language questions with source citations—all running completely offline on your machine with zero external API keys. The query endpoint streams Server-Sent Events and combines semantic and exact-keyword retrieval.

---

## 🏗️ Architecture & Tech Stack
- **Backend API:** FastAPI (Async REST API)
- **Local LLM Runtime:** Ollama (`phi3:mini` for text generation + `nomic-embed-text` for vector embeddings)
- **Vector Database:** LanceDB (Embedded serverless vector store)
- **Document Processing:** PyPDF, python-docx, pandas/openpyxl, JSON and structure-aware chunking
- **Retrieval:** LanceDB vector search fused with BM25 lexical search using reciprocal-rank fusion
- **Frontend:** Single-page glassmorphic UI built with Tailwind CSS & Vanilla JavaScript

---

## 🚀 Getting Started

### Prerequisites
1. Pull the required models (the bundled Ollama server is started and stopped with the app):
   ```bash
   ollama pull nomic-embed-text
   ollama pull phi3:mini
   ```
2. Python 3.11+ installed on your machine.

### Installation & Execution

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/local-rag-document-qa.git
   cd local-rag-document-qa
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI backend:

   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

5. Open your browser and navigate to **http://127.0.0.1:8000**.

The application starts `ollama\ollama.exe serve` only when the backend starts
and stops all local Ollama processes when the backend exits. Any Ollama process
already running is stopped before the project starts, so it cannot keep using
the GPU in the background. Set `OLLAMA_EXECUTABLE` if Ollama is installed
somewhere else. Stop the backend normally with `Ctrl+C` so its shutdown cleanup
can run.

---

## 📤 Push to GitHub

Open your terminal in the project folder and run:

```bash
git init
git add .
git commit -m "Initial commit: Production-ready local RAG pipeline with FastAPI, LanceDB, and Ollama"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/your-repo-name.git
git push -u origin main
```

Replace `YOUR_USERNAME/your-repo-name` with your actual GitHub repository URL.

---

## ✅ Updates Included

The current version keeps the original local, private RAG workflow and adds the
following production-oriented improvements:

### Backend and API

- Added FastAPI application lifespan management.
- Added automatic startup and shutdown of the bundled local Ollama server.
- Existing Ollama processes are stopped before the application starts so they
  do not continue using the GPU in the background.
- Added `GET /health` with Ollama, model, database, disk, document-count, and
  latency information.
- Added `POST /upload` with filename sanitization, extension validation, a
  50 MB upload limit, cleanup on failure, and clear HTTP errors.
- Added `GET /documents` to list indexed documents and chunk counts.
- Added `DELETE /documents/{source}` to remove a document and its stored file.
- Added streamed `POST /query` responses using Server-Sent Events.
- Added query-result caching with bounded in-memory storage.
- Added configurable similarity filtering and concurrency limits.

### Document ingestion

- Added support for PDF, DOCX, XLSX, CSV, JSON, Markdown, Python, JavaScript,
  TypeScript, and plain-text files.
- Added PDF extraction through `pypdf`.
- Added DOCX paragraph and table extraction through `python-docx`.
- Added spreadsheet and CSV extraction through `pandas` and `openpyxl`.
- Added formatted JSON extraction.
- Added binary-file detection for text and code uploads.
- Added structure-aware chunking with configurable overlap.
- Added deterministic chunk identifiers and safe replacement of re-uploaded
  documents.

### Retrieval and generation

- Added LanceDB vector retrieval with cosine-similarity fallback.
- Added BM25 lexical retrieval for exact terms, filenames, and identifiers.
- Added reciprocal-rank fusion to combine semantic and lexical results.
- Added source metadata, chunk indexes, similarity scores, and snippets to
  query events.
- Added grounded-answer prompting that restricts answers to retrieved context.
- Added an explicit fallback response when uploaded documents do not contain
  the requested information.
- Added low-temperature local generation with bounded output length.

### Frontend

- Added a refreshed glass-style single-page interface.
- Added drag-and-drop and multi-file upload support.
- Added document listing, chunk counts, and document deletion controls.
- Added live service health and query-latency indicators.
- Added streamed answer rendering with source citation chips.
- Added safe text escaping before rendering user or model content.

### Reliability and configuration

- Added retry handling for transient embedding requests.
- Added bounded Ollama concurrency using an asynchronous semaphore.
- Added configurable environment variables:

  ```text
  OLLAMA_EXECUTABLE
  OLLAMA_HOST
  OLLAMA_MAX_CONCURRENCY
  EMBEDDING_MODEL
  LLM_MODEL
  LANCEDB_PATH
  LANCEDB_TABLE_NAME
  MIN_SIMILARITY
  ```

- The default LanceDB path is anchored to the project directory rather than
  the process's current working directory.
- Bundled Ollama model discovery checks both supported local bundle layouts.
- Added `test_suite.py`, a standalone HTTP smoke/E2E test runner covering
  health, uploads, grounded queries, fallback responses, listing, and deletion.

### Validation

The updated source has been validated with Python compilation, whitespace
checks, and local smoke checks for chunking, binary detection, database access,
and project-relative path handling. Run the optional HTTP smoke suite against a
running server with:

```bash
python test_suite.py
```
