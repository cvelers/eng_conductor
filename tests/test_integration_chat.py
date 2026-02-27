import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings


def test_end_to_end_chat_has_citations() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.load().with_overrides(
        project_root=root,
        orchestrator_provider="mock",
        search_provider="mock",
        orchestrator_api_key="",
        search_api_key="",
        document_registry_path=root / "data" / "document_registry.json",
        tool_registry_path=root / "tools" / "tool_registry.json",
    )

    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "message": "Given S355, IPE 300, L=6 m, what is the bending resistance? Assume typical parameters if missing and list them."
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["supported"] is True
    assert "References:" in body["answer"] or "EN 1993-1-1" in body["answer"]
    assert len(body["sources"]) > 0
    assert any("6.2.5" in source["clause_id"] for source in body["sources"])
    assert "gamma_M0" in body["assumed_inputs"]


def test_ipe_moment_query_uses_ipe_tool_without_errors() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.load().with_overrides(
        project_root=root,
        orchestrator_provider="mock",
        search_provider="mock",
        orchestrator_api_key="",
        search_api_key="",
        document_registry_path=root / "data" / "document_registry.json",
        tool_registry_path=root / "tools" / "tool_registry.json",
    )

    app = create_app(settings)
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"message": "give me moment resistance of ipe 300"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["supported"] is True
    assert body["tool_trace"]
    assert body["tool_trace"][0]["tool_name"] == "ipe_moment_resistance_ec3"
    assert all(step["status"] == "ok" for step in body["tool_trace"])


def test_chat_writes_orchestrator_thread_log(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    log_path = tmp_path / "orchestrator_threads.json"
    settings = Settings.load().with_overrides(
        project_root=root,
        orchestrator_provider="mock",
        search_provider="mock",
        orchestrator_api_key="",
        search_api_key="",
        document_registry_path=root / "data" / "document_registry.json",
        tool_registry_path=root / "tools" / "tool_registry.json",
        orchestrator_thread_log_path=log_path,
    )

    app = create_app(settings)
    client = TestClient(app)

    response = client.post("/api/chat", json={"message": "find me shear resistance of ipe400"})
    assert response.status_code == 200

    assert log_path.exists()
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload

    entry = payload[-1]
    assert entry["endpoint"] == "/api/chat"
    assert entry["request"]["message"] == "find me shear resistance of ipe400"
    assert isinstance(entry["machine_events"], list)
    assert entry["response"] and entry["response"]["answer"]
    assert entry["error"] is None
