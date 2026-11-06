"""LanceDB persistence and hybrid vector/lexical retrieval."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("LANCEDB_PATH", str(PROJECT_DIR / "data" / "lancedb"))
TABLE_NAME = os.getenv("LANCEDB_TABLE_NAME", "document_chunks")


class DocumentChunk(BaseModel):
    id: str
    text: str
    source: str
    chunk_index: int = Field(ge=0)
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class LanceDBManager:
    def __init__(self, db_path: str = DB_PATH, table_name: str = TABLE_NAME):
        self.db_path = str(Path(db_path))
        self.table_name = table_name
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.table = self._open_or_create()

    def _open_or_create(self):
        if self.table_name in self.db.table_names():
            return self.db.open_table(self.table_name)
        schema = pa.schema([
            pa.field("id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("source", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("vector", pa.list_(pa.float32()), nullable=False),
            pa.field("metadata", pa.string(), nullable=False),
        ])
        return self.db.create_table(self.table_name, schema=schema)

    @staticmethod
    def _row(chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "id": chunk.id, "text": chunk.text, "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "vector": [float(value) for value in chunk.vector],
            "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
        }

    def insert_chunks(self, chunks: list[DocumentChunk], batch_size: int = 128) -> int:
        inserted = 0
        for start in range(0, len(chunks), batch_size):
            batch = [self._row(chunk) for chunk in chunks[start:start + batch_size]]
            if batch:
                self.table.add(batch)
                inserted += len(batch)
        return inserted

    @staticmethod
    def _decode_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}

    def _records(self) -> list[dict[str, Any]]:
        try:
            records = self.table.to_arrow().to_pylist()
        except Exception:
            records = self.table.to_pandas().to_dict("records")
        for record in records:
            record["metadata"] = self._decode_metadata(record.get("metadata"))
        return records

    def search_vector(self, query_vector: list[float], limit: int = 20) -> list[dict[str, Any]]:
        query = np.asarray(query_vector, dtype=np.float32)
        if query.size == 0:
            return []
        try:
            native = self.table.search(query.tolist()).limit(limit).to_list()
            results = []
            for row in native:
                item = dict(row)
                distance = max(0.0, float(item.get("_distance", item.get("distance", 1.0))))
                item["similarity"] = 1.0 / (1.0 + distance)
                item["metadata"] = self._decode_metadata(item.get("metadata"))
                results.append(item)
            if results:
                return results
        except Exception as exc:
            logger.warning("Native vector search unavailable: %s", exc)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        scored = []
        for row in self._records():
            vector = np.asarray(row.get("vector", []), dtype=np.float32)
            row_norm = np.linalg.norm(vector)
            if vector.size != query.size or row_norm == 0:
                continue
            score = float(np.dot(vector, query) / (row_norm * norm))
            scored.append((max(0.0, min(1.0, (score + 1.0) / 2.0)), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**row, "similarity": score} for score, row in scored[:limit]]

    def search_lexical(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        records = self._records()
        if not records:
            return []
        def tokenize(value: str) -> list[str]:
            return re.findall(r"[a-z0-9_./:-]+", value.lower())
        bm25 = BM25Okapi([tokenize(str(row.get("text", ""))) for row in records])
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [{**records[index], "lexical_score": float(score)} for index, score in ranked[:limit] if score > 0]

    def search_hybrid(self, query: str, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        vector_rows = self.search_vector(query_vector, max(limit * 4, 20))
        lexical_rows = self.search_lexical(query, max(limit * 4, 20))
        fused: dict[str, dict[str, Any]] = {}
        for rank, row in enumerate(vector_rows, 1):
            fused.setdefault(row["id"], {**row})["vector_rank"] = rank
        for rank, row in enumerate(lexical_rows, 1):
            fused.setdefault(row["id"], {**row}).update({"lexical_rank": rank, "lexical_score": row.get("lexical_score", 0)})
        for row in fused.values():
            row["rrf_score"] = (1 / (60 + row.get("vector_rank", 10000))) + (1 / (60 + row.get("lexical_rank", 10000)))
            query_terms = set(re.findall(r"[a-z0-9_./:-]+", query.lower()))
            text_terms = set(re.findall(r"[a-z0-9_./:-]+", str(row.get("text", "")).lower()))
            row["lexical_overlap"] = len(query_terms & text_terms) / max(1, len(query_terms))
        ranked = sorted(fused.values(), key=lambda row: row["rrf_score"], reverse=True)[:limit]
        for row in ranked:
            if not row.get("similarity") and row.get("lexical_score", 0) > 0:
                row["similarity"] = 0.5
            row["similarity"] = float(row.get("similarity", 0.0))
        return ranked

    def list_sources(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for row in self._records():
            counts[str(row.get("source", ""))] = counts.get(str(row.get("source", "")), 0) + 1
        return [{"source": source, "chunk_count": count} for source, count in sorted(counts.items())]

    def delete_by_source(self, source: str) -> int:
        count = sum(1 for row in self._records() if row.get("source") == source)
        if count:
            self.table.delete(f"source = '{source.replace(chr(39), chr(39) * 2)}'")
        return count

    def get_count(self) -> int:
        return int(self.table.count_rows())


db_manager = LanceDBManager()
