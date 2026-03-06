from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.auth import create_auth_router, require_auth
from backend.config import Settings
from backend.threads import create_threads_router
from backend.llm.factory import get_orchestrator_provider, get_search_provider, get_tool_writer_provider
from backend.logging_config import configure_logging
from backend.orchestrator.agent_loop import AgentLoop
from backend.orchestrator.core import CentralIntelligenceOrchestrator
from backend.registries.document_registry import load_all_clauses, load_document_registry
from backend.registries.tool_registry import load_tool_registry
from backend.retrieval.agentic_search import AgenticRetriever
from backend.schemas import ChatRequest, ChatResponse
from backend.tools.runner import MCPToolRunner
from backend.tools.writer import ToolWriter

logger = logging.getLogger(__name__)


def _append_thread_log(settings: Settings, payload: dict) -> None:
    log_path = settings.resolved_orchestrator_thread_log_path
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []

        if log_path.exists():
            raw = log_path.read_text(encoding="utf-8").strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        entries = [item for item in parsed if isinstance(item, dict)]
                    elif isinstance(parsed, dict):
                        entries = [parsed]
                except Exception:
                    # Backward compatibility: migrate old JSONL content into JSON array.
                    migrated: list[dict] = []
                    for line in raw.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(item, dict):
                            migrated.append(item)
                    entries = migrated

        entries.append(payload)
        log_path.write_text(json.dumps(entries, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("thread_log_write_failed", extra={"path": str(log_path)})


def _history_payload(history: list) -> list[dict]:
    rows: list[dict] = []
    for item in history or []:
        if hasattr(item, "model_dump"):
            rows.append(item.model_dump())
        elif isinstance(item, dict):
            rows.append(dict(item))
        else:
            rows.append({"role": str(getattr(item, "role", "")), "content": str(getattr(item, "content", ""))})
    return rows


class ToolGenerateRequest(BaseModel):
    description: str


def create_app(settings: Settings | None = None) -> FastAPI:
    load_dotenv()
    active_settings = settings or Settings.load()

    configure_logging(active_settings.log_level)

    doc_registry = load_document_registry(active_settings.resolved_document_registry_path)
    clauses = load_all_clauses(active_settings.project_root, doc_registry)
    tool_registry = load_tool_registry(active_settings.resolved_tool_registry_path)

    search_provider = get_search_provider(active_settings)
    orchestrator_provider = get_orchestrator_provider(active_settings)

    retriever = AgenticRetriever(
        settings=active_settings,
        search_provider=search_provider,
        clauses=clauses,
    )

    tool_runner = MCPToolRunner(project_root=active_settings.project_root, registry=tool_registry)

    orchestrator = CentralIntelligenceOrchestrator(
        settings=active_settings,
        orchestrator_llm=orchestrator_provider,
        retriever=retriever,
        tool_runner=tool_runner,
        tool_registry=tool_registry,
        document_registry=doc_registry,
        clauses=clauses,
    )

    agent_loop = AgentLoop(
        orchestrator=orchestrator,
        settings=active_settings,
    )

    tool_writer_provider = get_tool_writer_provider(active_settings)

    tool_writer = ToolWriter(
        llm=tool_writer_provider,
        retriever=retriever,
        tool_registry=orchestrator.tool_registry,
        project_root=active_settings.project_root,
    )

    app = FastAPI(title="Eurocodes Chatbot", version="0.3.0")
    app.state.orchestrator = orchestrator
    app.state.agent_loop = agent_loop
    app.state.tool_writer = tool_writer
    app.state.settings = active_settings

    app.include_router(create_auth_router(active_settings))
    app.include_router(create_threads_router(active_settings))
    auth_dep = require_auth(active_settings)

    frontend_dir = active_settings.project_root / "frontend"
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        request_id = uuid4().hex
        all_events: list[dict] = []
        final_response: ChatResponse | None = None
        error_detail: str | None = None
        try:
            for event_type, payload in agent_loop.run_stream(
                request.message,
                history=request.history,
                thinking_mode=request.thinking_mode,
                attachments=request.attachments,
                is_edit=request.is_edit,
            ):
                if event_type == "response":
                    final_response = payload
                elif isinstance(payload, dict):
                    all_events.append({"event_type": event_type, **payload})
            if final_response is None:
                raise RuntimeError("No response generated.")
            return final_response
        except Exception as exc:
            error_detail = str(exc)
            logger.exception("chat_failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            _append_thread_log(
                active_settings,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "request_id": request_id,
                    "endpoint": "/api/chat",
                    "request": {
                        "message": request.message,
                        "history": _history_payload(request.history),
                        "thinking_mode": request.thinking_mode,
                        "is_edit": request.is_edit,
                    },
                    "events": all_events,
                    "response": final_response.model_dump() if final_response else None,
                    "error": error_detail,
                },
            )

    @app.post("/api/chat/stream")
    async def chat_stream(request: ChatRequest) -> StreamingResponse:

        async def _stream_generator():
            request_id = uuid4().hex
            all_events: list[dict] = []
            final_response: ChatResponse | None = None
            error_detail: str | None = None
            try:
                for event_type, payload in agent_loop.run_stream(
                    request.message,
                    history=request.history,
                    thinking_mode=request.thinking_mode,
                    attachments=request.attachments,
                    is_edit=request.is_edit,
                ):
                    if event_type == "response":
                        final_response = payload
                        yield json.dumps({"type": "final", "response": payload.model_dump()}) + "\n"
                    elif event_type == "machine":
                        all_events.append(payload)
                        yield json.dumps({"type": "machine", **payload}) + "\n"
                    elif isinstance(payload, dict):
                        all_events.append({"event_type": event_type, **payload})
                        yield json.dumps({"type": event_type, **payload}) + "\n"
                    else:
                        yield json.dumps({"type": event_type, "content": str(payload)}) + "\n"
                    await asyncio.sleep(0.005)
            except Exception as exc:
                error_detail = str(exc)
                logger.exception("chat_stream_failed")
                yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            else:
                if final_response is None:
                    error_detail = "No response generated."
                    yield json.dumps({"type": "error", "detail": "No response generated."}) + "\n"
            finally:
                _append_thread_log(
                    active_settings,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "request_id": request_id,
                        "endpoint": "/api/chat/stream",
                        "request": {
                            "message": request.message,
                            "history": _history_payload(request.history),
                            "thinking_mode": request.thinking_mode,
                            "is_edit": request.is_edit,
                        },
                        "events": all_events,
                        "response": final_response.model_dump() if final_response else None,
                        "error": error_detail,
                    },
                )

        return StreamingResponse(_stream_generator(), media_type="application/x-ndjson")

    @app.post("/api/tools/generate")
    async def generate_tool(request: ToolGenerateRequest):
        try:
            result = tool_writer.generate(request.description)
            return result
        except Exception as exc:
            logger.exception("tool_generation_failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/tools")
    async def list_tools():
        return [
            {
                "tool_name": entry.tool_name,
                "description": entry.description,
                "tags": entry.tags,
            }
            for entry in tool_registry
        ]

    return app


app = create_app()
