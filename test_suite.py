"""Standalone smoke/E2E test runner for a running local RAG server."""

from __future__ import annotations

import io
import json
import sys

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def make_pdf(text: str) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(text) + 35} >>\nstream\nBT /F1 12 Tf 72 720 Td ({text}) Tj ET\nendstream".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode())
    return bytes(output)


def main() -> int:
    checks = []
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        files = {
            "facts.txt": b"Project Atlas has a blue deployment window on Friday at 18:00.",
            "facts.csv": io.BytesIO(("key,value\nowner,Platform Team\nregion,Local\n").encode()),
            "facts.json": json.dumps({"service": "Atlas", "tier": "gold"}).encode(),
        }
        files["facts.pdf"] = make_pdf("Atlas PDF notes: deployment window Friday at 18:00.")
        try:
            health = client.get("/health")
            checks.append(report("GET /health", health.is_success, health.text[:160]))
            uploaded = []
            for name, content in files.items():
                response = client.post("/upload", files={"file": (name, content)})
                data = response.json()
                ok = response.is_success and int(data.get("chunks_processed", 0)) > 0
                checks.append(report(f"Upload {name}", ok, response.text[:160]))
                if ok:
                    uploaded.append(name)
            query = client.post("/query", json={"question": "When is the Atlas deployment window?"})
            events = [json.loads(line[6:]) for line in query.text.splitlines() if line.startswith("data: ")]
            answer = "".join(event.get("token", "") for event in events)
            sources = next((event.get("sources", []) for event in events if event.get("type") == "metadata"), [])
            checks.append(report("Grounded streaming query", bool(query.is_success and answer and sources), answer[:120]))
            missing = client.post("/query", json={"question": "What is the launch date for Project Neptune?"})
            missing_events = [json.loads(line[6:]) for line in missing.text.splitlines() if line.startswith("data: ")]
            missing_answer = "".join(event.get("token", "") for event in missing_events)
            checks.append(report("Fallback grounding", bool(missing.is_success and "cannot find" in missing_answer.lower())))
            docs = client.get("/documents").json()
            checks.append(report("Document listing", len(docs) >= len(uploaded)))
            if uploaded:
                deletion = client.delete(f"/documents/{uploaded[0]}")
                checks.append(report("Document deletion", deletion.is_success))
                for source in uploaded[1:]:
                    client.delete(f"/documents/{source}")
                remaining = {item["source"] for item in client.get("/documents").json()}
                checks.append(report("Deletion cleanup", not set(uploaded) & remaining))
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            checks.append(report("Test execution", False, str(exc)))
    passed = sum(checks)
    print(f"\nTest report: {passed}/{len(checks)} passed")
    return 0 if checks and passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
