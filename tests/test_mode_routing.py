from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.orchestrator.engine import CentralIntelligenceOrchestrator
from backend.registries.document_registry import ClauseRecord
from backend.registries.tool_registry import ToolRegistryEntry
from backend.retrieval.agentic_search import RetrievedClause


class RoutingLLM(LLMProvider):
    provider_name = "routing-test"

    def __init__(
        self,
        *,
        mode: str,
        plan_tools: list[str] | None = None,
        post_tools: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.plan_tools = plan_tools or []
        self.post_tools = post_tools or []
        self.calls: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        prompt = f"{system_prompt}\n{user_prompt}".lower()
        if "###task:plan###" in prompt:
            self.calls.append("plan")
            return json.dumps(
                {
                    "mode": self.mode,
                    "tools": self.plan_tools,
                    "rationale": "routing-test plan",
                }
            )

        if "###task:plan_tools###" in prompt:
            self.calls.append("plan_tools")
            return json.dumps(
                {
                    "tools": self.post_tools,
                    "rationale": "routing-test post retrieval plan",
                }
            )

        if "###task:extract_inputs###" in prompt:
            self.calls.append("extract")
            return json.dumps(
                {
                    "user_inputs": {},
                    "assumed_inputs": {"section_name": "IPE300", "steel_grade": "S355", "gamma_M0": 1.0},
                    "assumptions": ["gamma_M0 = 1.00"],
                    "tool_inputs": {},
                }
            )

        self.calls.append("compose")
        return "Routing test answer."


class TrackingRetriever:
    def __init__(self, sequence: list[str]) -> None:
        self.calls = 0
        self.sequence = sequence

    def iter_retrieve(self, query: str, top_k: int | None = None):
        self.calls += 1
        self.sequence.append("retrieval")
        clause = ClauseRecord(
            doc_id="ec3.en1993-1-1.2005",
            doc_title="EN 1993-1-1",
            standard="EN 1993-1-1",
            clause_id="6.2.5",
            clause_title="Bending moment",
            text="Clause text",
            keywords=["bending"],
            pointer="en_1993_1_1_2005#/6.2.5",
        )
        retrieved = RetrievedClause(clause=clause, score=10.0, matched_terms=["bending"])
        yield {
            "type": "final",
            "results": [retrieved],
            "trace": [
                {
                    "iteration": 1,
                    "query": query,
                    "top_clause_ids": ["ec3.en1993-1-1.2005:6.2.5"],
                }
            ],
        }


class TrackingToolRunner:
    def __init__(self, sequence: list[str]) -> None:
        self.calls: list[str] = []
        self.sequence = sequence

    def run(self, tool_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(tool_name)
        self.sequence.append(f"tool:{tool_name}")
        return {
            "status": "ok",
            "result": {
                "inputs_used": dict(inputs),
                "outputs": {"M_Rd_kNm": 123.0},
                "notes": [],
                "clause_references": [
                    {
                        "doc_id": "ec3.en1993-1-1.2005",
                        "clause_id": "6.2.5",
                        "title": "Bending moment",
                        "pointer": "en_1993_1_1_2005#/6.2.5",
                    }
                ],
            },
        }


def _tool_entry(name: str) -> ToolRegistryEntry:
    return ToolRegistryEntry(
        tool_name=name,
        description=f"{name} test tool",
        script_path="tools/mcp/runner.py",
        input_schema={"type": "object", "properties": {"section_name": {"type": "string"}}},
        output_schema={"type": "object"},
    )


def _make_orchestrator(
    *,
    llm: RoutingLLM,
    retriever: TrackingRetriever,
    runner: TrackingToolRunner,
) -> CentralIntelligenceOrchestrator:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.load().with_overrides(project_root=root, top_k_clauses=3)
    return CentralIntelligenceOrchestrator(
        settings=settings,
        orchestrator_llm=llm,
        retriever=retriever,  # type: ignore[arg-type]
        tool_runner=runner,  # type: ignore[arg-type]
        tool_registry=[
            _tool_entry("section_classification_ec3"),
            _tool_entry("member_resistance_ec3"),
            _tool_entry("interaction_check_ec3"),
        ],
    )


def test_calculator_mode_skips_retrieval_and_runs_tools() -> None:
    sequence: list[str] = []
    llm = RoutingLLM(mode="calculator", plan_tools=["member_resistance_ec3"])
    retriever = TrackingRetriever(sequence)
    runner = TrackingToolRunner(sequence)
    orchestrator = _make_orchestrator(llm=llm, retriever=retriever, runner=runner)

    response = orchestrator.run("calculate moment resistance for ipe300")

    assert retriever.calls == 0
    assert runner.calls == ["section_classification_ec3", "member_resistance_ec3"]
    assert response.retrieval_trace == []
    assert response.tool_trace


def test_retrieval_only_mode_runs_no_tools() -> None:
    sequence: list[str] = []
    llm = RoutingLLM(mode="retrieval_only", plan_tools=["member_resistance_ec3"])
    retriever = TrackingRetriever(sequence)
    runner = TrackingToolRunner(sequence)
    orchestrator = _make_orchestrator(llm=llm, retriever=retriever, runner=runner)

    response = orchestrator.run("explain section classification procedure")

    assert retriever.calls == 1
    assert runner.calls == []
    assert response.retrieval_trace
    assert response.tool_trace == []


def test_hybrid_mode_refines_tools_after_retrieval() -> None:
    sequence: list[str] = []
    llm = RoutingLLM(
        mode="hybrid",
        plan_tools=[],
        post_tools=["member_resistance_ec3"],
    )
    retriever = TrackingRetriever(sequence)
    runner = TrackingToolRunner(sequence)
    orchestrator = _make_orchestrator(llm=llm, retriever=retriever, runner=runner)

    response = orchestrator.run("check bending resistance and verify procedure")

    assert retriever.calls == 1
    assert runner.calls == ["section_classification_ec3", "member_resistance_ec3"]
    assert llm.calls.count("plan_tools") == 1
    assert sequence.index("retrieval") < sequence.index("tool:section_classification_ec3")
    assert response.tool_trace
