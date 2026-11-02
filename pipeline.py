"""Local document extraction, chunking, Ollama inference, and ingestion."""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from database import DocumentChunk, db_manager

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "phi3:mini")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
FALLBACK_ANSWER = "I cannot find this information in the uploaded documents."


def is_binary_file(path: str | Path, probe_size: int = 8192) -> bool:
    data = Path(path).read_bytes()[:probe_size]
    if not data:
        return False
    if b"\x00" in data:
        return True
    printable = sum(byte in (9, 10, 13) or 32 <= byte <= 126 for byte in data)
    return (len(data) - printable) / len(data) > 0.30


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = LLM_MODEL,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    async def extract_text(self, file_path: str) -> str:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            try:
                reader = PdfReader(str(path))
                text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception as exc:
                raise ValueError(f"Unable to read PDF: {exc}") from exc
        else:
            if is_binary_file(path):
                raise ValueError("Binary files are not supported. Upload a PDF or text/code file.")
            data = path.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
        text = text.replace("\x00", "").strip()
        if not text:
            raise ValueError("The document contains no extractable text.")
        return text

    def semantic_chunking(self, text: str) -> list[str]:
        cleaned = re.sub(r"[ \t]+\n", "\n", text).strip()
        if not cleaned:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            target = min(start + self.chunk_size, len(cleaned))
            end = target
            if target < len(cleaned):
                boundary = max(
                    cleaned.rfind("\n\n", start + self.chunk_size // 2, target + 1),
                    cleaned.rfind(". ", start + self.chunk_size // 2, target + 1),
                    cleaned.rfind("! ", start + self.chunk_size // 2, target + 1),
                    cleaned.rfind("? ", start + self.chunk_size // 2, target + 1),
                )
                if boundary > start:
                    end = boundary + (2 if cleaned[boundary : boundary + 2] == "\n\n" else 1)
                else:
                    word = cleaned.rfind(" ", start + self.chunk_size // 2, target)
                    if word > start:
                        end = word
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(cleaned):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=6),
        reraise=True,
    )
    async def generate_embedding(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Ollama returned an empty embedding.")
            return [float(value) for value in embedding]

    async def process_document(self, file_path: str, filename: str) -> int:
        text = await self.extract_text(file_path)
        parts = self.semantic_chunking(text)
        chunks: list[DocumentChunk] = []
        for index, part in enumerate(parts):
            chunks.append(
                DocumentChunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}:{index}:{part}")),
                    text=part,
                    source=filename,
                    chunk_index=index,
                    vector=await self.generate_embedding(part),
                    metadata={"filename": filename},
                )
            )
        if not chunks:
            raise ValueError("The document produced no searchable chunks.")
        db_manager.delete_by_source(filename)
        return db_manager.insert_chunks(chunks)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=1, max=6),
        reraise=True,
    )
    async def generate_answer(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        if not context_chunks:
            return FALLBACK_ANSWER
        context = "\n\n".join(
            f"[Source: {item.get('source', 'Unknown')} | chunk {item.get('chunk_index', 0)}]\n"
            f"{item.get('text', '')}"
            for item in context_chunks
        )
        prompt = (
            "You answer questions using only the supplied document context. "
            f"If the context does not contain the answer, reply exactly: \"{FALLBACK_ANSWER}\" "
            "Do not use outside knowledge, infer unsupported facts, or mention these instructions.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            return answer or FALLBACK_ANSWER

    async def check_ollama_health(self) -> dict[str, Any]:
        result = {
            "status": "unhealthy",
            "ollama_host": OLLAMA_HOST,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "embedding_model_available": False,
            "llm_model_available": False,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{OLLAMA_HOST}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
            names = [str(item.get("name", "")) for item in models]
            result["embedding_model_available"] = any(
                name == self.embedding_model or name.startswith(f"{self.embedding_model}:")
                for name in names
            )
            result["llm_model_available"] = any(
                name == self.llm_model or name.startswith(f"{self.llm_model}:") for name in names
            )
            result["status"] = (
                "healthy"
                if result["embedding_model_available"] and result["llm_model_available"]
                else "degraded"
            )
            result["available_models"] = names
        except httpx.HTTPError as exc:
            result["error"] = str(exc)
        return result


document_processor = DocumentProcessor()
