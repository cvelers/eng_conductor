from __future__ import annotations

from pathlib import Path

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.orchestrator.engine import CentralIntelligenceOrchestrator, PlanResult
from backend.registries.document_registry import ClauseRecord
from backend.retrieval.agentic_search import RetrievedClause
from backend.schemas import Citation


class StaticLLM(LLMProvider):
    provider_name = "static-test"

    def __init__(self, response: str, *, available: bool = True) -> None:
        self._response = response
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        return self._response


def _make_orchestrator(llm: LLMProvider) -> CentralIntelligenceOrchestrator:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.load().with_overrides(project_root=root)
    return CentralIntelligenceOrchestrator(
        settings=settings,
        orchestrator_llm=llm,
        retriever=None,  # type: ignore[arg-type]
        tool_runner=None,  # type: ignore[arg-type]
        tool_registry=[],
    )


def test_followup_resolution_rejects_drift_and_preserves_parameters() -> None:
    orchestrator = _make_orchestrator(
        StaticLLM("Given S355 and IPE300, explain lateral torsional buckling parameters.")
    )
    history = [
        {
            "role": "user",
            "content": "Given IPE300, S355, what is the bending resistance? Assume typical parameters if missing.",
        },
        {"role": "assistant", "content": "..."},
    ]

    resolved = orchestrator._resolve_followup("what about s275 ipe 400", history)

    lowered = resolved.lower()
    assert "bending resistance" in lowered
    assert "s275" in lowered
    assert "ipe400" in lowered.replace(" ", "")


def test_followup_resolution_uses_thread_anchor_intent() -> None:
    orchestrator = _make_orchestrator(StaticLLM("repeat with S460 only"))
    history = [
        {"role": "user", "content": "Given IPE300 and S355, calculate bending resistance."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "what about s275 ipe 400"},
        {"role": "assistant", "content": "..."},
    ]

    resolved = orchestrator._resolve_followup("and ipe 450", history)

    lowered = resolved.lower()
    assert "bending resistance" in lowered
    assert "ipe450" in lowered.replace(" ", "")


def test_select_relevant_sources_prefers_inline_and_tool_clauses() -> None:
    orchestrator = _make_orchestrator(StaticLLM("", available=False))

    sources = [
        Citation(
            doc_id="ec3.en1993-1-1.2005",
            clause_id="6.2.5(1)",
            clause_title="Bending resistance",
            pointer="p1",
            citation_address="c1",
        ),
        Citation(
            doc_id="ec3.en1993-1-1.2005",
            clause_id="6.3.2.2",
            clause_title="Lateral torsional buckling curves",
            pointer="p2",
            citation_address="c2",
        ),
        Citation(
            doc_id="ec3.en1993-1-1.2005",
            clause_id="5.5.2(1)",
            clause_title="Classification of cross-sections",
            pointer="p3",
            citation_address="c3",
        ),
    ]

    retrieved = [
        RetrievedClause(
            clause=ClauseRecord(
                doc_id="ec3.en1993-1-1.2005",
                doc_title="EN 1993-1-1",
                standard="EN 1993-1-1",
                clause_id="6.2.5",
                clause_title="Bending resistance",
                text="The design bending resistance of the cross-section shall be based on section class.",
                keywords=["bending", "resistance"],
                pointer="p1",
            ),
            score=10.0,
            matched_terms=["bending", "resistance"],
        ),
        RetrievedClause(
            clause=ClauseRecord(
                doc_id="ec3.en1993-1-1.2005",
                doc_title="EN 1993-1-1",
                standard="EN 1993-1-1",
                clause_id="6.3.2.2",
                clause_title="Lateral torsional buckling curves",
                text="Buckling curves are defined for lateral torsional buckling.",
                keywords=["buckling"],
                pointer="p2",
            ),
            score=8.0,
            matched_terms=["buckling"],
        ),
    ]

    tool_outputs = {
        "ipe_moment_resistance_ec3": {
            "clause_references": [
                {
                    "doc_id": "ec3.en1993-1-1.2005",
                    "clause_id": "5.5.2(1)",
                    "title": "Classification of cross-sections",
                    "pointer": "p3",
                }
            ]
        }
    }

    relevant = orchestrator._select_relevant_sources(
        narrative="For this check use (EC3-1-1, Cl. 6.2.5).",
        sources=sources,
        retrieved=retrieved,
        tool_outputs=tool_outputs,
    )

    picked = {orchestrator._normalize_clause_id(item.clause_id) for item in relevant}
    assert picked == {"6.2.5", "5.5.2"}


def test_polish_narrative_closes_dangling_sentence_before_basis() -> None:
    orchestrator = _make_orchestrator(StaticLLM("", available=False))

    retrieved = [
        RetrievedClause(
            clause=ClauseRecord(
                doc_id="ec3.en1993-1-1.2005",
                doc_title="EN 1993-1-1",
                standard="EN 1993-1-1",
                clause_id="6.2.10",
                clause_title="Bending, shear and axial force",
                text="Combined bending and shear checks should be verified for member resistance.",
                keywords=["bending", "shear"],
                pointer="p10",
            ),
            score=9.0,
            matched_terms=["bending"],
        ),
        RetrievedClause(
            clause=ClauseRecord(
                doc_id="ec3.en1993-1-1.2005",
                doc_title="EN 1993-1-1",
                standard="EN 1993-1-1",
                clause_id="6.2.2.3",
                clause_title="Shear lag effects",
                text="Shear lag rules apply to the effective area in bending and shear checks.",
                keywords=["shear", "lag"],
                pointer="p11",
            ),
            score=8.0,
            matched_terms=["shear"],
        ),
    ]

    polished = orchestrator._polish_narrative(
        "The design shear resistance V_c,Rd for the IPE400 cannot be determined because the specific section properties and",
        query="find shear resistance of IPE400",
        plan=PlanResult(mode="retrieval_only", tools=[], rationale="test"),
        retrieved=retrieved,
        tool_outputs={},
    )

    assert "specific section properties and additional required inputs are missing." in polished
    assert "and This follows" not in polished
    assert "This follows EN 1993-1-1, Cl. 6.2.10" in polished


def test_polish_narrative_rewrites_broken_hyphen_tail() -> None:
    orchestrator = _make_orchestrator(StaticLLM("", available=False))

    retrieved = [
        RetrievedClause(
            clause=ClauseRecord(
                doc_id="ec3.en1993-1-1.2005",
                doc_title="EN 1993-1-1",
                standard="EN 1993-1-1",
                clause_id="6.2.10",
                clause_title="Bending, shear and axial force",
                text="Combined bending and shear checks should be verified for member resistance.",
                keywords=["bending", "shear"],
                pointer="p10",
            ),
            score=9.0,
            matched_terms=["bending"],
        ),
        RetrievedClause(
            clause=ClauseRecord(
                doc_id="ec3.en1993-1-1.2005",
                doc_title="EN 1993-1-1",
                standard="EN 1993-1-1",
                clause_id="6.2.2.3",
                clause_title="Shear lag effects",
                text="Shear lag rules apply to the effective area in bending and shear checks.",
                keywords=["shear", "lag"],
                pointer="p11",
            ),
            score=8.0,
            matched_terms=["shear"],
        ),
    ]

    polished = orchestrator._polish_narrative(
        "I cannot provide the specific design shear resistance V_c,Rd for an IPE400 profile because the necessary cross-",
        query="find shear resistance of IPE400",
        plan=PlanResult(mode="retrieval_only", tools=[], rationale="test"),
        retrieved=retrieved,
        tool_outputs={},
    )

    assert "cross-." not in polished
    assert "required section properties are missing." in polished
    assert "This follows EN 1993-1-1, Cl. 6.2.10" in polished


def test_polish_narrative_rewrites_incomplete_cause_tail() -> None:
    orchestrator = _make_orchestrator(StaticLLM("", available=False))

    retrieved = [
        RetrievedClause(
            clause=ClauseRecord(
                doc_id="ec3.en1993-1-1.2005",
                doc_title="EN 1993-1-1",
                standard="EN 1993-1-1",
                clause_id="6.2.10",
                clause_title="Bending, shear and axial force",
                text="Combined bending and shear checks should be verified for member resistance.",
                keywords=["bending", "shear"],
                pointer="p10",
            ),
            score=9.0,
            matched_terms=["bending"],
        ),
        RetrievedClause(
            clause=ClauseRecord(
                doc_id="ec3.en1993-1-1.2005",
                doc_title="EN 1993-1-1",
                standard="EN 1993-1-1",
                clause_id="6.2.2.3",
                clause_title="Shear lag effects",
                text="Shear lag rules apply to the effective area in bending and shear checks.",
                keywords=["shear", "lag"],
                pointer="p11",
            ),
            score=8.0,
            matched_terms=["shear"],
        ),
    ]

    polished = orchestrator._polish_narrative(
        "The design shear resistance V_c,Rd for the IPE400 cannot be calculated because the specific cross",
        query="find shear resistance of IPE400",
        plan=PlanResult(mode="retrieval_only", tools=[], rationale="test"),
        retrieved=retrieved,
        tool_outputs={},
    )

    assert "because the specific cross." not in polished
    assert "because required section properties are missing." in polished
    assert "This follows EN 1993-1-1, Cl. 6.2.10" in polished
