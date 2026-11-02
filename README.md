# 🔒 Local Document Q&A (Private RAG Pipeline)

A 100% local, private Retrieval-Augmented Generation (RAG) web application. Upload documents (PDF, TXT, Markdown, code), chunk and embed them using local Ollama models, store vectors in an embedded LanceDB database, and ask natural language questions with source citations—all running completely offline on your machine with zero external API keys.

---

## 🏗️ Architecture & Tech Stack
- **Backend API:** FastAPI (Async REST API)
- **Local LLM Runtime:** Ollama (`phi3:mini` for text generation + `nomic-embed-text` for vector embeddings)
- **Vector Database:** LanceDB (Embedded serverless vector store)
- **Document Processing:** PyPDF & custom semantic text chunking
- **Frontend:** Single-page glassmorphic UI built with Tailwind CSS & Vanilla JavaScript

---

## 🚀 Getting Started

### Prerequisites
1. [Ollama](https://ollama.com/) installed and running (`ollama serve`)
2. Pull the required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull phi3:mini
   ```
3. Python 3.11+ installed on your machine.

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
