"""Local extraction, structure-aware chunking, Ollama inference, and ingestion."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pandas as pd
from docx import Document
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from database import DocumentChunk, db_manager

logger = logging.getLogger(__name__)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "phi3:mini")
CHUNK_SIZE, CHUNK_OVERLAP = 512, 50
FALLBACK_ANSWER = "I cannot find this information in the uploaded documents."
TEXT_EXTENSIONS = {".md", ".py", ".js", ".ts", ".txt"}


def is_binary_file(path: str | Path, probe_size: int = 8192) -> bool:
    data = Path(path).read_bytes()[:probe_size]
    if not data:
        return False
    if b"\x00" in data:
        return True
    printable = sum(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in data)
    return (len(data) - printable) / len(data) > 0.30


class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
                 embedding_model: str = EMBEDDING_MODEL, llm_model: str = LLM_MODEL):
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.embedding_model, self.llm_model = embedding_model, llm_model
        self.ollama_host = OLLAMA_HOST
        self._ollama_slots = asyncio.Semaphore(
            max(1, int(os.getenv("OLLAMA_MAX_CONCURRENCY", "2")))
        )

    async def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                text = "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            elif suffix == ".docx":
                doc = Document(str(path))
                text = "\n\n".join(p.text for p in doc.paragraphs)
                text += "\n\n" + "\n".join(" | ".join(cell.text.strip() for cell in row.cells) for table in doc.tables for row in table.rows)
            elif suffix in {".xlsx", ".csv"}:
                sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl") if suffix == ".xlsx" else {"data": pd.read_csv(path)}
                text = "\n\n".join(f"## {name}\n{frame.fillna('').to_markdown(index=False)}" for name, frame in sheets.items())
            elif suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                text = json.dumps(value, ensure_ascii=False, indent=2)
            elif suffix in TEXT_EXTENSIONS:
                if is_binary_file(path):
                    raise ValueError("Binary files are not supported.")
                text = path.read_text(encoding="utf-8-sig")
            else:
                raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and str(exc) == "Binary files are not supported.":
                raise
            raise ValueError(f"Unable to read {path.name}: {exc}") from exc
        text = text.replace("\x00", "").strip()
        if not text:
            raise ValueError("The document contains no extractable text.")
        return text

    def semantic_chunking(self, text: str) -> list[str]:
        cleaned = re.sub(r"[ \t]+\n", "\n", text).strip()
        blocks = re.split(r"\n{2,}", cleaned)
        chunks, current = [], ""
        for block in blocks:
            candidate = f"{current}\n\n{block}".strip() if current else block
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = block
            while len(current) > self.chunk_size:
                cut = max((current.rfind(mark, self.chunk_size // 2, self.chunk_size + 1) for mark in (". ", "! ", "? ", "\n", " ")), default=-1)
                cut = cut + 1 if cut > 0 else self.chunk_size
                chunks.append(current[:cut].strip())
                current = current[max(0, cut - self.chunk_overlap):].strip()
        if current:
            chunks.append(current)
        return [chunk for chunk in chunks if chunk]

    @retry(retry=retry_if_exception_type((httpx.HTTPError, ValueError)), stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=0.5, min=1, max=6), reraise=True)
    async def generate_embedding(self, text: str) -> list[float]:
        async with self._ollama_slots:
            async with httpx.AsyncClient(timeout=90) as client:
                response = await client.post(f"{self.ollama_host}/api/embeddings", json={"model": self.embedding_model, "prompt": text})
                response.raise_for_status()
                embedding = response.json().get("embedding")
                if not isinstance(embedding, list) or not embedding:
                    raise ValueError("Ollama returned an empty embedding.")
                return [float(value) for value in embedding]

    async def process_document(self, file_path: str, filename: str) -> int:
        parts = self.semantic_chunking(await self.extract_text(file_path))
        chunks = [DocumentChunk(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{index}:{part}")), text=part,
            source=filename, chunk_index=index, vector=await self.generate_embedding(part),
            metadata={"filename": filename, "extension": Path(filename).suffix.lower()},
        ) for index, part in enumerate(parts)]
        if not chunks:
            raise ValueError("The document produced no searchable chunks.")
        db_manager.delete_by_source(filename)
        return db_manager.insert_chunks(chunks)

    def _prompt(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        context = "\n\n".join(f"[{item.get('source', 'Unknown')}#{item.get('chunk_index', 0)}]\n{item.get('text', '')}" for item in context_chunks)
        return (f"Answer only from this context. If missing, reply exactly: {FALLBACK_ANSWER}\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")

    async def stream_answer(self, question: str, context_chunks: list[dict[str, Any]]) -> AsyncIterator[str]:
        async with self._ollama_slots:
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream("POST", f"{self.ollama_host}/api/generate",
                                        json={"model": self.llm_model, "prompt": self._prompt(question, context_chunks),
                                              "stream": True, "options": {"temperature": 0.1, "num_predict": 512}}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        payload = json.loads(line)
                        token = str(payload.get("response", ""))
                        if token:
                            yield token

    async def check_ollama_health(self) -> dict[str, Any]:
        result = {"status": "unhealthy", "ollama_host": self.ollama_host, "embedding_model": self.embedding_model,
                  "llm_model": self.llm_model, "embedding_model_available": False, "llm_model_available": False}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                response.raise_for_status()
                names = [str(item.get("name", "")) for item in response.json().get("models", [])]
            result["embedding_model_available"] = any(name == self.embedding_model or name.startswith(f"{self.embedding_model}:") for name in names)
            result["llm_model_available"] = any(name == self.llm_model or name.startswith(f"{self.llm_model}:") for name in names)
            result["status"] = "healthy" if result["embedding_model_available"] and result["llm_model_available"] else "degraded"
        except (httpx.HTTPError, ValueError, OSError) as exc:
            result["error"] = str(exc)
        return result


document_processor = DocumentProcessor()
