"""Own the lifetime of the bundled Ollama server."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import psutil

logger = logging.getLogger(__name__)


class OllamaManager:
    def __init__(self, host: str, project_dir: Path):
        self.host = host
        self.project_dir = project_dir
        self.process: asyncio.subprocess.Process | None = None
        self.started_by_app = False

    @property
    def executable(self) -> Path:
        configured = os.getenv("OLLAMA_EXECUTABLE")
        return Path(configured) if configured else self.project_dir / "ollama" / "ollama.exe"

    async def _is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(f"{self.host}/api/tags")
                return response.is_success
        except httpx.HTTPError:
            return False

    @staticmethod
    def _is_ollama_process(process: psutil.Process) -> bool:
        try:
            name = (process.name() or "").lower()
            return name.startswith("ollama") or name in {"llama-server.exe", "ollama_llama_server.exe"}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    async def _stop_existing_processes(self) -> None:
        if urlparse(self.host).hostname not in {"127.0.0.1", "localhost", "::1"}:
            return

        processes = [process for process in psutil.process_iter(["name"]) if self._is_ollama_process(process)]
        if processes:
            logger.info("Stopping %s existing Ollama process(es) before starting the project.", len(processes))
        for process in processes:
            try:
                children = process.children(recursive=True)
                for child in children:
                    child.kill()
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if processes:
            await asyncio.to_thread(psutil.wait_procs, processes, timeout=5)

    async def start(self) -> dict[str, Any]:
        executable = self.executable
        if not executable.is_file():
            message = (
                f"Ollama is unavailable at {self.host} and the bundled executable was not found at "
                f"{executable}. Set OLLAMA_EXECUTABLE or start Ollama manually."
            )
            logger.error(message)
            raise RuntimeError(message)

        await self._stop_existing_processes()
        environment = os.environ.copy()
        environment["OLLAMA_HOST"] = self.host
        bundled_model_dirs = (
            self.project_dir / "ollama" / "models",
            self.project_dir / "ollama" / "lib" / "ollama" / "models",
        )
        for models_dir in bundled_model_dirs:
            if models_dir.is_dir():
                environment.setdefault("OLLAMA_MODELS", str(models_dir))
                break

        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = await asyncio.create_subprocess_exec(
            str(executable),
            "serve",
            cwd=str(executable.parent),
            env=environment,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creation_flags,
        )
        self.started_by_app = True
        for _ in range(30):
            if await self._is_available():
                logger.info("Started Ollama for this application (pid %s).", self.process.pid)
                return {"status": "started", "host": self.host, "started_by_app": True, "pid": self.process.pid}
            if self.process.returncode is not None:
                error = (await self.process.stderr.read()).decode(errors="replace").strip()
                raise RuntimeError(f"Ollama exited during startup: {error or 'unknown error'}")
            await asyncio.sleep(0.5)

        await self.stop()
        raise RuntimeError(f"Ollama did not become available at {self.host} within 15 seconds.")

    async def stop(self) -> None:
        if not self.process or not self.started_by_app:
            return
        process = self.process
        self.process = None
        self.started_by_app = False
        if process.returncode is not None:
            await self._stop_existing_processes()
            return
        logger.info("Stopping Ollama process started by this application (pid %s).", process.pid)
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        await self._stop_existing_processes()
