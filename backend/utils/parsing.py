from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.utils.json_utils import parse_json_loose, strip_code_fences

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"\b((?:IPE|HEA|HEB|HEM)\s*\d{2,4})\b", re.IGNORECASE)
_STEEL_RE = re.compile(r"\b(S(?:235|275|355|420|460))\b", re.IGNORECASE)


@dataclass
class ExtractionResult:
    user_inputs: dict[str, Any]
    assumed_inputs: dict[str, Any]
    assumptions: list[str]
    tool_inputs: dict[str, dict[str, Any]]


_EXTRACTION_SYSTEM = """\
You are an engineering input extractor for Eurocode 3 calculations.
Given a user query and a set of tool schemas, extract all explicitly stated values \
and identify what defaults should be assumed for missing required inputs.

Return JSON only with this exact shape:
{
  "user_inputs": { ... values the user explicitly stated ... },
  "assumed_inputs": { ... reasonable defaults for values the user did NOT state ... },
  "assumptions": [ "human-readable note for each assumed value" ],
  "tool_inputs": {
    "tool_name_1": { ... complete input dict ready for this tool ... },
    "tool_name_2": { ... complete input dict ready for this tool ... }
  }
}

Rules:
- Normalize units: lengths to meters, forces to kN, moments to kNm, stresses to MPa.
- Section names: uppercase, no spaces (e.g. "IPE300", "HEA200").
- Steel grades: "S235", "S275", "S355", "S420", "S460".
- Only include tools listed in planned_tools.
- For tool_inputs, merge user values with reasonable defaults to create complete ready-to-run inputs.
- If a tool depends on outputs from a previous tool in the chain, use null for those fields.
- Be conservative with assumptions — note each one clearly.
- Infer load_type from context (e.g. "UDL" → "udl", "point load at midspan" → "point_mid")."""


def extract_inputs(
    *,
    query: str,
    planned_tools: list[str],
    tool_registry: dict[str, Any],
    llm: LLMProvider,
    settings: Settings,
) -> ExtractionResult:
    if not planned_tools:
        return ExtractionResult(
            user_inputs={}, assumed_inputs={}, assumptions=[], tool_inputs={}
        )

    if not llm.available:
        return _fallback_extraction(planned_tools, settings, query=query)

    tool_schemas: dict[str, Any] = {}
    for name in planned_tools:
        entry = tool_registry.get(name)
        if entry:
            tool_schemas[name] = {
                "description": entry.description,
                "input_schema": entry.input_schema,
                "constraints": entry.constraints,
                "examples": entry.examples[:2],
            }

    prompt = (
        "###TASK:EXTRACT_INPUTS###\n"
        f"User query: {query}\n\n"
        f"Planned tools (in execution order): {json.dumps(planned_tools)}\n\n"
        f"Tool schemas:\n{json.dumps(tool_schemas, indent=2)}\n\n"
        f"Default values when user does not specify:\n"
        f"- steel_grade: {settings.default_steel_grade}\n"
        f"- section_name: {settings.default_section_name}\n"
        f"- gamma_M0: {settings.default_gamma_m0}\n"
        f"- MEd_kNm: {settings.default_med_knm}\n"
        f"- NEd_kN: {settings.default_ned_kn}\n\n"
        "Extract all inputs from the query and fill defaults. Return JSON only."
    )

    try:
        raw = llm.generate(
            system_prompt=_EXTRACTION_SYSTEM,
            user_prompt=prompt,
            temperature=0,
            max_tokens=1400,
        )
        parsed = parse_json_loose(raw)
        return ExtractionResult(
            user_inputs=parsed.get("user_inputs", {}),
            assumed_inputs=parsed.get("assumed_inputs", {}),
            assumptions=parsed.get("assumptions", []),
            tool_inputs=parsed.get("tool_inputs", {}),
        )
    except Exception as exc:
        logger.warning("llm_input_extraction_failed", extra={"error": str(exc)})
        return _fallback_extraction(planned_tools, settings, query=query)


def _strip_code_fences(text: str) -> str:
    # Backward-compat shim for existing call sites/tests.
    return strip_code_fences(text)


def _fallback_extraction(
    planned_tools: list[str], settings: Settings, *, query: str = "",
) -> ExtractionResult:
    """Minimal fallback when LLM is unavailable."""
    normalized_query = query or ""
    section_match = _SECTION_RE.search(normalized_query)
    steel_match = _STEEL_RE.search(normalized_query)

    parsed_section = (
        section_match.group(1).replace(" ", "").upper()
        if section_match
        else settings.default_section_name
    )
    parsed_steel = steel_match.group(1).upper() if steel_match else settings.default_steel_grade

    user_inputs: dict[str, Any] = {}
    assumptions: list[str] = []
    if section_match:
        user_inputs["section_name"] = parsed_section
    else:
        assumptions.append(
            f"Section assumed as {settings.default_section_name} (LLM extraction unavailable)."
        )
    if steel_match:
        user_inputs["steel_grade"] = parsed_steel
    else:
        assumptions.append(
            f"Steel grade assumed as {settings.default_steel_grade} (LLM extraction unavailable)."
        )

    assumed: dict[str, Any] = {
        "steel_grade": parsed_steel,
        "section_name": parsed_section,
        "gamma_M0": settings.default_gamma_m0,
    }
    assumptions.append(f"γ_M0 = {settings.default_gamma_m0:.2f} (LLM extraction unavailable).")

    tool_inputs: dict[str, dict[str, Any]] = {}
    for name in planned_tools:
        tool_inputs[name] = {
            "section_name": parsed_section,
            "steel_grade": parsed_steel,
            "gamma_M0": settings.default_gamma_m0,
        }

    return ExtractionResult(
        user_inputs=user_inputs,
        assumed_inputs=assumed,
        assumptions=assumptions,
        tool_inputs=tool_inputs,
    )
