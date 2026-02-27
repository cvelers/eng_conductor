from __future__ import annotations

import json
import re

from backend.llm.base import LLMProvider

_IPE_RE = re.compile(r"ipe\s*(\d+)", re.IGNORECASE)


class MockProvider(LLMProvider):
    provider_name = "mock"

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
            return self._mock_plan(prompt)

        if "###task:extract_inputs###" in prompt:
            return self._mock_extract(prompt)

        if "###task:decompose###" in prompt:
            return self._mock_decompose(prompt)

        if "###task:relevance###" in prompt:
            return self._mock_relevance(prompt)

        if "###task:gap###" in prompt:
            return json.dumps(None)

        if "###task:refine###" in prompt:
            return json.dumps(["ec3 section classification", "en 1993-1-1 bending resistance"])

        if "follow-up" in prompt or "expanded self-contained query" in prompt:
            return "Mock response."

        return "Grounded summary: The response is based only on retrieved EC3 clauses and tool outputs."

    @staticmethod
    def _extract_user_query(prompt: str) -> str:
        if "user query:" in prompt:
            return prompt.split("user query:")[1].split("\n")[0].strip()
        return prompt

    def _mock_plan(self, prompt: str) -> str:
        q = self._extract_user_query(prompt)

        if "ipe" in q and ("moment" in q or "m_rd" in q) and "interaction" not in q:
            return json.dumps({
                "mode": "hybrid",
                "tools": ["ipe_moment_resistance_ec3"],
                "rationale": "Mock: IPE moment resistance query.",
            })

        mode = "retrieval_only"
        tools: list[str] = []

        if any(t in q for t in ["column buckling", "flexural buckling"]):
            mode = "hybrid"
            tools = ["column_buckling_ec3"]
        elif any(t in q for t in ["bolt", "m20", "m16"]) and "shear" in q:
            mode = "hybrid"
            tools = ["bolt_shear_ec3"]
        elif "cantilever" in q:
            mode = "hybrid"
            tools = ["cantilever_beam_calculator"]
        elif any(t in q for t in ["simply supported", "simple beam", "udl"]):
            mode = "hybrid"
            tools = ["simple_beam_calculator"]
        elif any(t in q for t in ["resistance", "given", "check", "m_ed", "n_ed"]):
            mode = "hybrid"
            tools = ["section_classification_ec3", "member_resistance_ec3"]

        if "interaction" in q or ("combined" in q and any(t in q for t in ["bending", "axial"])):
            if "interaction_check_ec3" not in tools:
                tools.append("interaction_check_ec3")

        return json.dumps({
            "mode": mode, "tools": tools,
            "rationale": "Mock deterministic plan.",
        })

    def _mock_extract(self, prompt: str) -> str:
        q = self._extract_user_query(prompt)

        user_inputs: dict = {}
        assumed_inputs: dict = {}
        assumptions: list[str] = []
        tool_inputs: dict = {}

        section = "IPE300"
        steel = "S355"
        gamma = 1.0

        ipe_match = _IPE_RE.search(q)
        if ipe_match:
            section = f"IPE{ipe_match.group(1)}"
            user_inputs["section_name"] = section

        for grade in ["s235", "s275", "s355", "s420", "s460"]:
            if grade in q:
                steel = grade.upper()
                user_inputs["steel_grade"] = steel
                break

        if "section_name" not in user_inputs:
            assumed_inputs["section_name"] = section
            assumptions.append(f"Section assumed as {section}.")
        if "steel_grade" not in user_inputs:
            assumed_inputs["steel_grade"] = steel
            assumptions.append(f"Steel grade assumed as {steel}.")
        assumed_inputs["gamma_M0"] = gamma
        assumptions.append(f"γ_M0 = {gamma:.2f} (standard EC3 value).")

        if "ipe_moment_resistance_ec3" in prompt:
            tool_inputs["ipe_moment_resistance_ec3"] = {
                "section_name": user_inputs.get("section_name", section),
                "steel_grade": user_inputs.get("steel_grade", steel),
                "section_class": 2,
                "gamma_M0": gamma,
            }

        if "section_classification_ec3" in prompt:
            tool_inputs["section_classification_ec3"] = {
                "section_name": user_inputs.get("section_name", section),
                "steel_grade": user_inputs.get("steel_grade", steel),
            }

        if "member_resistance_ec3" in prompt:
            tool_inputs["member_resistance_ec3"] = {
                "section_name": user_inputs.get("section_name", section),
                "steel_grade": user_inputs.get("steel_grade", steel),
                "section_class": None,
                "gamma_M0": gamma,
            }

        if "interaction_check_ec3" in prompt:
            tool_inputs["interaction_check_ec3"] = {
                "MEd_kNm": 120.0,
                "NEd_kN": 200.0,
                "M_Rd_kNm": None,
                "N_Rd_kN": None,
            }

        return json.dumps({
            "user_inputs": user_inputs,
            "assumed_inputs": assumed_inputs,
            "assumptions": assumptions,
            "tool_inputs": tool_inputs,
        })

    def _mock_decompose(self, prompt: str) -> str:
        base = prompt.split("query:")[1].split("\n")[0].strip() if "query:" in prompt else "ec3 clause"
        return json.dumps([base, f"{base} formula", f"{base} table"])

    def _mock_relevance(self, prompt: str) -> str:
        scores = []
        idx = 1
        while f"{idx}." in prompt:
            scores.append({"idx": idx, "score": max(10 - idx, 3)})
            idx += 1
        return json.dumps(scores) if scores else json.dumps([{"idx": 1, "score": 8}])
