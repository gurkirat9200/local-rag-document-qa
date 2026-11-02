"""FastAPI entry point for the local document Q&A application."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import db_manager
from pipeline import FALLBACK_ANSWER, document_processor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MAX_FILE_SIZE = 50 * 1024 * 1024
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Local Document Q&A", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    ollama = await document_processor.check_ollama_health()
    try:
        chunks = db_manager.get_count()
        database = {"status": "healthy", "document_chunks": chunks}
    except Exception as exc:
        database = {"status": "unhealthy", "error": str(exc), "document_chunks": 0}
    status = "healthy" if database["status"] == "healthy" and ollama["status"] == "healthy" else "degraded"
    if ollama["status"] == "unhealthy" or database["status"] == "unhealthy":
        status = "unhealthy"
    return {"status": status, "database": database, "ollama": ollama, "documents_stored": chunks if database["status"] == "healthy" else 0}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    if not filename or filename in {".", ".."}:
        raise HTTPException(400, "A valid filename is required.")
    destination = UPLOAD_DIR / filename
    written = 0
    try:
        async with aiofiles.open(destination, "wb") as target:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if written > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large. Maximum size is 50 MB.")
                await target.write(block)
        count = await document_processor.process_document(str(destination), filename)
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
        raise HTTPException(500, "Document processing failed.") from exc


@app.post("/query")
async def query(request: QueryRequest) -> dict[str, Any]:
    try:
        vector = await document_processor.generate_embedding(request.question)
        matches = [
            item
            for item in db_manager.search_similar(vector, request.limit)
            if float(item.get("similarity", 0.0)) >= 0.35
        ]
        if not matches:
            return {"answer": FALLBACK_ANSWER, "sources": []}
        answer = await document_processor.generate_answer(request.question, matches)
        sources = [
            {
                "source": item.get("source", "Unknown"),
                "chunk_index": int(item.get("chunk_index", 0)),
                "similarity": round(float(item.get("similarity", 0.0)), 4),
                "text_snippet": str(item.get("text", ""))[:240],
            }
            for item in matches
        ]
        return {"answer": answer, "sources": sources}
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(502, "Unable to query the local AI service.") from exc


@app.get("/documents", response_model=list[SourceInfo])
async def documents() -> list[SourceInfo]:
    return [SourceInfo(**item) for item in db_manager.list_sources()]


@app.delete("/documents/{source}")
async def delete_document(source: str) -> dict[str, Any]:
    count = db_manager.delete_by_source(source)
    if not count:
        raise HTTPException(404, "Document not found.")
    (UPLOAD_DIR / Path(source).name).unlink(missing_ok=True)
    return {"message": "Document deleted.", "source": source, "chunks_deleted": count}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
