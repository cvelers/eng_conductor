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
_SPAN_RE = re.compile(
    r"\b(?:span|l)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*m\b|\b(\d+(?:\.\d+)?)\s*m\s*span\b",
    re.IGNORECASE,
)
_UDL_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:kN|kn)\s*/\s*m\b|\b(\d+(?:\.\d+)?)\s*(?:kN|kn)\s*per\s*m\b",
    re.IGNORECASE,
)
_POINT_LOAD_RE = re.compile(
    r"\b(?:point(?:\s+load)?|load)\s*(?:of|=)?\s*(\d+(?:\.\d+)?)\s*(?:kN|kn)\b"
    r"|\b(\d+(?:\.\d+)?)\s*(?:kN|kn)\s*point\b",
    re.IGNORECASE,
)
_POSITION_RE = re.compile(
    r"\b(?:at|position(?:_a)?|a)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*m\b",
    re.IGNORECASE,
)
_E_GPA_RE = re.compile(r"\be\s*[=:]?\s*(\d+(?:\.\d+)?)\s*gpa\b", re.IGNORECASE)
_I_CM4_RE = re.compile(r"\bi\s*[=:]?\s*(\d+(?:\.\d+)?)\s*cm4\b", re.IGNORECASE)


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
            max_tokens=4096,
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
    """Fallback extraction when LLM is unavailable.

    Uses lightweight regex parsing and tool-aware defaults so required tool
    inputs remain valid even during provider outages.
    """
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
    assumed: dict[str, Any] = {}
    assumptions: list[str] = []

    span_m = _parse_number(_SPAN_RE, normalized_query)
    udl_kn_per_m = _parse_number(_UDL_RE, normalized_query)
    point_kn = _parse_number(_POINT_LOAD_RE, normalized_query)
    position_a_m = _parse_number(_POSITION_RE, normalized_query)
    e_gpa = _parse_number(_E_GPA_RE, normalized_query)
    i_cm4 = _parse_number(_I_CM4_RE, normalized_query)
    lowered_query = normalized_query.lower()

    inferred_load_type = "point_mid"
    if any(token in lowered_query for token in ("udl", "uniform", "distributed")):
        inferred_load_type = "udl"
    elif "point" in lowered_query:
        inferred_load_type = "point_mid" if "mid" in lowered_query else "point"

    def mark_user(key: str, value: Any) -> None:
        if value is not None:
            user_inputs[key] = value

    def mark_assumed(key: str, value: Any, note: str) -> None:
        if key not in user_inputs:
            assumed[key] = value
            assumptions.append(note)

    ec3_base = {
        "section_name": parsed_section,
        "steel_grade": parsed_steel,
        "gamma_M0": settings.default_gamma_m0,
    }
    tool_inputs: dict[str, dict[str, Any]] = {}
    for name in planned_tools:
        if name in {"simple_beam_calculator", "cantilever_beam_calculator"}:
            tool_payload: dict[str, Any] = {}

            mark_user("span_m", span_m)
            if span_m is not None:
                tool_payload["span_m"] = span_m
            else:
                tool_payload["span_m"] = 6.0
                mark_assumed("span_m", 6.0, "Span assumed as 6.0 m (LLM extraction unavailable).")

            mark_user("load_type", inferred_load_type)
            tool_payload["load_type"] = inferred_load_type

            if inferred_load_type == "udl":
                mark_user("load_kn_per_m", udl_kn_per_m)
                if udl_kn_per_m is not None:
                    tool_payload["load_kn_per_m"] = udl_kn_per_m
                else:
                    tool_payload["load_kn_per_m"] = 10.0
                    mark_assumed(
                        "load_kn_per_m",
                        10.0,
                        "UDL assumed as 10.0 kN/m (LLM extraction unavailable).",
                    )
            else:
                mark_user("load_kn", point_kn)
                if point_kn is not None:
                    tool_payload["load_kn"] = point_kn
                else:
                    tool_payload["load_kn"] = 50.0
                    mark_assumed(
                        "load_kn",
                        50.0,
                        "Point load assumed as 50.0 kN (LLM extraction unavailable).",
                    )

                if inferred_load_type == "point":
                    mark_user("position_a_m", position_a_m)
                    if position_a_m is not None:
                        tool_payload["position_a_m"] = position_a_m

            if e_gpa is not None:
                mark_user("E_gpa", e_gpa)
                tool_payload["E_gpa"] = e_gpa
            else:
                tool_payload["E_gpa"] = 210.0
                mark_assumed("E_gpa", 210.0, "Assumed Young's modulus E = 210 GPa for steel.")

            if i_cm4 is not None:
                mark_user("I_cm4", i_cm4)
                tool_payload["I_cm4"] = i_cm4

            tool_inputs[name] = tool_payload
            continue

        if section_match:
            mark_user("section_name", parsed_section)
        else:
            mark_assumed(
                "section_name",
                settings.default_section_name,
                f"Section assumed as {settings.default_section_name} (LLM extraction unavailable).",
            )
        if steel_match:
            mark_user("steel_grade", parsed_steel)
        else:
            mark_assumed(
                "steel_grade",
                settings.default_steel_grade,
                f"Steel grade assumed as {settings.default_steel_grade} (LLM extraction unavailable).",
            )
        mark_assumed(
            "gamma_M0",
            settings.default_gamma_m0,
            f"γ_M0 = {settings.default_gamma_m0:.2f} (LLM extraction unavailable).",
        )
        tool_inputs[name] = dict(ec3_base)

    return ExtractionResult(
        user_inputs=user_inputs,
        assumed_inputs=assumed,
        assumptions=assumptions,
        tool_inputs=tool_inputs,
    )


def _parse_number(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if not match:
        return None
    for group in match.groups():
        if group is None:
            continue
        try:
            return float(group)
        except ValueError:
            continue
    return None
