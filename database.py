"""LanceDB persistence for document chunks."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("LANCEDB_PATH", "./data/lancedb")
TABLE_NAME = os.getenv("LANCEDB_TABLE_NAME", "document_chunks")


class DocumentChunk(BaseModel):
    id: str
    text: str
    source: str
    chunk_index: int = Field(ge=0)
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class LanceDBManager:
    """Owns the LanceDB connection and normalizes its version differences."""

    def __init__(self, db_path: str = DB_PATH, table_name: str = TABLE_NAME):
        self.db_path = str(Path(db_path))
        self.table_name = table_name
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.table = self._open_or_create()

    def _open_or_create(self):
        names = self.db.table_names()
        if self.table_name in names:
            return self.db.open_table(self.table_name)
        schema = pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("text", pa.string(), nullable=False),
                pa.field("source", pa.string(), nullable=False),
                pa.field("chunk_index", pa.int32(), nullable=False),
                pa.field("vector", pa.list_(pa.float32()), nullable=False),
                pa.field("metadata", pa.string(), nullable=False),
            ]
        )
        return self.db.create_table(self.table_name, schema=schema)

    @staticmethod
    def _row(chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "text": chunk.text,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "vector": [float(value) for value in chunk.vector],
            "metadata": json.dumps(chunk.metadata, ensure_ascii=False),
        }

    def insert_chunks(self, chunks: list[DocumentChunk], batch_size: int = 128) -> int:
        """Insert chunks in bounded batches and return the inserted count."""
        inserted = 0
        for start in range(0, len(chunks), batch_size):
            batch = [self._row(chunk) for chunk in chunks[start : start + batch_size]]
            if batch:
                self.table.add(batch)
                inserted += len(batch)
        return inserted

    def _records(self) -> list[dict[str, Any]]:
        try:
            records = self.table.to_arrow().to_pylist()
        except Exception:
            records = self.table.to_pandas().to_dict("records")
        for record in records:
            metadata = record.get("metadata", "{}")
            if isinstance(metadata, str):
                try:
                    record["metadata"] = json.loads(metadata)
                except json.JSONDecodeError:
                    record["metadata"] = {}
        return records

    def search_similar(
        self, query_vector: list[float], limit: int = 5, filter_source: str | None = None
    ) -> list[dict[str, Any]]:
        """Search natively, with a cosine-distance fallback for incompatible versions."""
        query = np.asarray(query_vector, dtype=np.float32)
        if query.size == 0:
            return []
        try:
            search = self.table.search(query.tolist())
            if filter_source:
                escaped = filter_source.replace("'", "''")
                search = search.where(f"source = '{escaped}'")
            native = search.limit(limit).to_list()
            results = [dict(row) for row in native]
            for row in results:
                distance = max(0.0, float(row.get("_distance", row.get("distance", 1.0))))
                row["similarity"] = 1.0 / (1.0 + distance)
                row["metadata"] = self._decode_metadata(row.get("metadata"))
            if results:
                return results
        except Exception as exc:
            logger.warning("Native LanceDB search failed; using cosine fallback: %s", exc)

        scored: list[tuple[float, dict[str, Any]]] = []
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        for row in self._records():
            if filter_source and row.get("source") != filter_source:
                continue
            vector = np.asarray(row.get("vector", []), dtype=np.float32)
            norm = np.linalg.norm(vector)
            if vector.size != query.size or norm == 0:
                continue
            score = float(np.dot(vector, query) / (norm * query_norm))
            scored.append((max(0.0, min(1.0, (score + 1.0) / 2.0)), row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{**row, "similarity": score} for score, row in scored[:limit]]

    @staticmethod
    def _decode_metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            return json.loads(value or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}

    def list_sources(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for row in self._records():
            source = str(row.get("source", ""))
            counts[source] = counts.get(source, 0) + 1
        return [
            {"source": source, "chunk_count": count}
            for source, count in sorted(counts.items(), key=lambda item: item[0].lower())
        ]

    def delete_by_source(self, source: str) -> int:
        before = sum(1 for row in self._records() if row.get("source") == source)
        if before:
            escaped = source.replace("'", "''")
            self.table.delete(f"source = '{escaped}'")
        return before

    def get_count(self) -> int:
        return int(self.table.count_rows())


db_manager = LanceDBManager()
