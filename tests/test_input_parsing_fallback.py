from __future__ import annotations

from pathlib import Path

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.registries.tool_registry import load_tool_registry
from backend.utils.parsing import extract_inputs


class UnavailableLLM(LLMProvider):
    provider_name = "unavailable-test"

    @property
    def available(self) -> bool:
        return False

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> str:
        raise RuntimeError("LLM should not be called when unavailable.")


def _tool_map(root: Path) -> dict:
    entries = load_tool_registry(root / "tools" / "tool_registry.json")
    return {entry.tool_name: entry for entry in entries}


def test_fallback_extracts_simple_beam_inputs_without_llm() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.load().with_overrides(project_root=root)
    registry = _tool_map(root)

    result = extract_inputs(
        query="Simply supported beam, 6m span, 15 kN/m UDL. What is the maximum bending moment and deflection?",
        planned_tools=["simple_beam_calculator"],
        tool_registry=registry,
        llm=UnavailableLLM(),
        settings=settings,
    )

    tool_payload = result.tool_inputs["simple_beam_calculator"]
    assert tool_payload["span_m"] == 6.0
    assert tool_payload["load_type"] == "udl"
    assert tool_payload["load_kn_per_m"] == 15.0
    assert "section_name" not in tool_payload
    assert "steel_grade" not in tool_payload
    assert result.user_inputs["span_m"] == 6.0
    assert result.user_inputs["load_kn_per_m"] == 15.0


def test_fallback_keeps_ec3_defaults_for_member_tools() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings.load().with_overrides(project_root=root)
    registry = _tool_map(root)

    result = extract_inputs(
        query="check moment resistance",
        planned_tools=["member_resistance_ec3"],
        tool_registry=registry,
        llm=UnavailableLLM(),
        settings=settings,
    )

    tool_payload = result.tool_inputs["member_resistance_ec3"]
    assert tool_payload["section_name"] == settings.default_section_name
    assert tool_payload["steel_grade"] == settings.default_steel_grade
    assert tool_payload["gamma_M0"] == settings.default_gamma_m0
