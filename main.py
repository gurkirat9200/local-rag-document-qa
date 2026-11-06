"""FastAPI application for private local document Q&A."""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from collections import OrderedDict
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import db_manager
from ollama_manager import OllamaManager
from pipeline import FALLBACK_ANSWER, document_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR, UPLOAD_DIR = BASE_DIR / "data", BASE_DIR / "data" / "uploads"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE, MIN_SIMILARITY = 50 * 1024 * 1024, float(os.getenv("MIN_SIMILARITY", "0.35"))
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".csv", ".json", ".md", ".py", ".js", ".ts", ".txt"}
QUERY_CACHE: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
CACHE_SIZE = 64
ollama_manager = OllamaManager(document_processor.ollama_host, BASE_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await ollama_manager.start()
    try:
        yield
    finally:
        await ollama_manager.stop()


app = FastAPI(title="Local Document Q&A", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


class SourceInfo(BaseModel):
    source: str
    chunk_count: int


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
async def health() -> dict[str, Any]:
    started = time.perf_counter()
    ollama = await document_processor.check_ollama_health()
    try:
        database = {"status": "healthy", "document_chunks": db_manager.get_count(), "table": db_manager.table_name}
    except Exception as exc:
        database = {"status": "unhealthy", "document_chunks": 0, "error": str(exc)}
    status = "healthy" if database["status"] == "healthy" and ollama["status"] == "healthy" else "degraded"
    disk = shutil.disk_usage(DATA_DIR)
    try:
        documents_stored = len(db_manager.list_sources())
    except Exception:
        documents_stored = 0
    return {"status": status, "database": database, "ollama": ollama,
            "disk_free_bytes": disk.free, "disk_total_bytes": disk.total,
            "documents_stored": documents_stored,
            "check_latency_ms": round((time.perf_counter() - started) * 1000, 2)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    if not filename or filename in {".", ".."}:
        raise HTTPException(400, "A valid filename is required.")
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, "Unsupported file type.")
    destination, written = UPLOAD_DIR / filename, 0
    try:
        async with aiofiles.open(destination, "wb") as target:
            while block := await file.read(1024 * 1024):
                written += len(block)
                if written > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large. Maximum size is 50 MB.")
                await target.write(block)
        count = await document_processor.process_document(str(destination), filename)
        QUERY_CACHE.clear()
        return {"message": "Document ingested successfully.", "filename": filename, "chunks_processed": count}
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except (ValueError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload failed for %s", filename)
        destination.unlink(missing_ok=True)
        raise HTTPException(502, "Document processing failed.") from exc


def _sources(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"source": item.get("source", "Unknown"), "chunk_index": int(item.get("chunk_index", 0)),
             "similarity": round(float(item.get("similarity", 0.0)), 4), "rrf_score": round(float(item.get("rrf_score", 0.0)), 6),
             "text_snippet": str(item.get("text", ""))[:240]} for item in matches]


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    started = time.perf_counter()
    key = (request.question.strip().lower(), request.limit)
    try:
        cached = QUERY_CACHE.get(key)
        if cached:
            QUERY_CACHE.move_to_end(key)
            async def cached_stream():
                yield f"data: {json.dumps({'type': 'metadata', 'sources': cached['sources'], 'cached': True})}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'token': cached['answer']})}\n\n"
                yield "data: {\"type\":\"done\"}\n\n"
            return StreamingResponse(cached_stream(), media_type="text/event-stream")
        vector = await document_processor.generate_embedding(request.question)
        query_terms = len(set(request.question.lower().split()))
        matches = [
            item for item in db_manager.search_hybrid(request.question, vector, request.limit)
            if float(item.get("similarity", 0.0)) >= MIN_SIMILARITY
            and (
                float(item.get("similarity", 0.0)) >= 0.62
                or float(item.get("lexical_overlap", 0.0)) >= min(1.0, 2 / max(1, query_terms))
            )
        ]
        sources = _sources(matches)

        async def event_stream():
            try:
                if not matches:
                    answer = FALLBACK_ANSWER
                    yield f"data: {json.dumps({'type': 'metadata', 'sources': [], 'latency_ms': round((time.perf_counter() - started) * 1000)})}\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'token': answer})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'metadata', 'sources': sources, 'latency_ms': round((time.perf_counter() - started) * 1000)})}\n\n"
                    answer_parts = []
                    async for token in document_processor.stream_answer(request.question, matches):
                        answer_parts.append(token)
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                    answer = "".join(answer_parts).strip() or FALLBACK_ANSWER
                QUERY_CACHE[key] = {"answer": answer, "sources": sources}
                QUERY_CACHE.move_to_end(key)
                while len(QUERY_CACHE) > CACHE_SIZE:
                    QUERY_CACHE.popitem(last=False)
            except Exception:
                logger.exception("Streaming query failed")
                yield f"data: {json.dumps({'type': 'error', 'message': 'Local AI generation failed.'})}\n\n"
            yield "data: {\"type\":\"done\"}\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(502, "Unable to query the local AI service.") from exc


@app.get("/documents", response_model=list[SourceInfo])
async def documents() -> list[SourceInfo]:
    return [SourceInfo(**item) for item in db_manager.list_sources()]


@app.delete("/documents/{source}")
async def delete_document(source: str) -> dict[str, Any]:
    source = Path(source).name
    count = db_manager.delete_by_source(source)
    if not count:
        raise HTTPException(404, "Document not found.")
    (UPLOAD_DIR / source).unlink(missing_ok=True)
    QUERY_CACHE.clear()
    return {"message": "Document deleted.", "source": source, "chunks_deleted": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
