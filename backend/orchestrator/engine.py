from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterator

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.registries.document_registry import ClauseRecord, DocumentRegistryEntry
from backend.registries.tool_registry import ToolRegistryEntry
from backend.retrieval.agentic_search import AgenticRetriever, RetrievedClause
from backend.schemas import ChatResponse, Citation, RetrievalTraceStep, ToolTraceStep
from backend.tools.runner import MCPToolRunner
from backend.utils.citations import build_citation_address
from backend.utils.json_utils import parse_json_loose
from backend.utils.parsing import ExtractionResult, extract_inputs

logger = logging.getLogger(__name__)

_UNIT_SUFFIXES = [
    ("_kNm", " (kNm)"),
    ("_kN", " (kN)"),
    ("_MPa", " (MPa)"),
    ("_GPa", " (GPa)"),
    ("_mm", " (mm)"),
    ("_cm2", " (cm²)"),
    ("_cm3", " (cm³)"),
    ("_cm4", " (cm⁴)"),
    ("_m", " (m)"),
]

_KEY_SUBSCRIPTS: dict[str, str] = {
    "M_Rd": "M<sub>Rd</sub>",
    "N_Rd": "N<sub>Rd</sub>",
    "V_Rd": "V<sub>Rd</sub>",
    "M_Ed": "M<sub>Ed</sub>",
    "N_Ed": "N<sub>Ed</sub>",
    "V_Ed": "V<sub>Ed</sub>",
    "N_b_Rd": "N<sub>b,Rd</sub>",
    "Fv_Rd": "F<sub>v,Rd</sub>",
    "Fv": "F<sub>v</sub>",
    "fy": "f<sub>y</sub>",
    "fu": "f<sub>u</sub>",
    "fub": "f<sub>ub</sub>",
    "Wpl": "W<sub>pl</sub>",
    "Wel": "W<sub>el</sub>",
    "L_cr": "L<sub>cr</sub>",
    "gamma_M0": "γ<sub>M0</sub>",
    "gamma_M1": "γ<sub>M1</sub>",
    "gamma_M2": "γ<sub>M2</sub>",
    "alpha_v": "α<sub>v</sub>",
}

_NARRATIVE_SUB_RE: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gamma_M([012])"), r"γ<sub>M\1</sub>"),
    (re.compile(r"γ_M([012])"), r"γ<sub>M\1</sub>"),
    (re.compile(r"γM([012])"), r"γ<sub>M\1</sub>"),
    (re.compile(r"N_b,Rd"), "N<sub>b,Rd</sub>"),
    (re.compile(r"Fv[_,]Rd"), "F<sub>v,Rd</sub>"),
    (re.compile(r"M_Rd"), "M<sub>Rd</sub>"),
    (re.compile(r"N_Rd"), "N<sub>Rd</sub>"),
    (re.compile(r"V_Rd"), "V<sub>Rd</sub>"),
    (re.compile(r"M_Ed"), "M<sub>Ed</sub>"),
    (re.compile(r"N_Ed"), "N<sub>Ed</sub>"),
    (re.compile(r"V_Ed"), "V<sub>Ed</sub>"),
    (re.compile(r"F_v\b"), "F<sub>v</sub>"),
    (re.compile(r"f_ub"), "f<sub>ub</sub>"),
    (re.compile(r"f_y"), "f<sub>y</sub>"),
    (re.compile(r"f_u\b"), "f<sub>u</sub>"),
    (re.compile(r"W_pl"), "W<sub>pl</sub>"),
    (re.compile(r"W_el"), "W<sub>el</sub>"),
    (re.compile(r"L_cr"), "L<sub>cr</sub>"),
    (re.compile(r"α_?v\b"), "α<sub>v</sub>"),
]


@dataclass
class PlanResult:
    mode: str
    tools: list[str]
    rationale: str


_CHAIN_MAPPINGS: dict[str, dict[str, tuple[str, str, Any]]] = {
    "member_resistance_ec3": {
        "section_class": ("section_classification_ec3", "outputs.governing_class", 2),
    },
    "interaction_check_ec3": {
        "M_Rd_kNm": ("member_resistance_ec3", "outputs.M_Rd_kNm", None),
        "N_Rd_kN": ("member_resistance_ec3", "outputs.N_Rd_kN", None),
    },
}

def _resolve_nested(data: dict[str, Any] | None, path: str) -> Any:
    """Walk a dot-separated path into a nested dict."""
    if data is None:
        return None
    current: Any = data
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


_FOLLOWUP_VALUE_RE = re.compile(
    r"\b(?:IPE\s*\d+|HEA\s*\d+|HEB\s*\d+|S(?:235|275|355|420|460)|M\d+|\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

_DANGLING_TAIL_RE = re.compile(
    r"(?:\b(?:and|or|with|because|for|to|of|in|on|at|from|that|which|where|when|if|while|as)\b[\s,;:]*)+$",
    re.IGNORECASE,
)

_BROKEN_HYPHEN_TAIL_RE = re.compile(r"(?:\b[0-9A-Za-z_]+-)$")
_INCOMPLETE_CAUSE_TAIL_RE = re.compile(
    r"(?:\b(?:because|due to|as)\b[^.?!]{0,160}?\b(?:cross|cross[- ]?section|section|properties?|inputs?|data|parameters?)\b)$",
    re.IGNORECASE,
)


class CentralIntelligenceOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        orchestrator_llm: LLMProvider,
        retriever: AgenticRetriever,
        tool_runner: MCPToolRunner,
        tool_registry: list[ToolRegistryEntry],
        document_registry: list[DocumentRegistryEntry] | None = None,
    ) -> None:
        self.settings = settings
        self.orchestrator_llm = orchestrator_llm
        self.retriever = retriever
        self.tool_runner = tool_runner
        self.tool_registry = {entry.tool_name: entry for entry in tool_registry}
        self.document_registry = document_registry or []

    def run(self, query: str, *, history: list | None = None) -> ChatResponse:
        final_response: ChatResponse | None = None
        for event_type, payload in self.run_stream(query, history=history):
            if event_type == "response":
                final_response = payload
        if final_response is None:
            raise RuntimeError("Orchestrator did not produce a final response.")
        return final_response

    def run_stream(self, raw_query: str, *, history: list | None = None) -> Iterator[tuple[str, Any]]:
        query = self._resolve_followup(raw_query, history or [])

        # --- INTAKE ---
        yield ("machine", {"node": "intake", "status": "active", "title": "Query Intake", "detail": "Analyzing your question..."})
        plan = self._build_plan(query)
        yield ("machine", {"node": "intake", "status": "done", "title": "Query Intake", "detail": "Question understood."})
        yield ("machine", {
            "node": "plan", "status": "done", "title": "Pathway Planning",
            "detail": f"Strategy: {plan.mode} | Tools: {plan.tools or ['none']}",
            "meta": {"mode": plan.mode, "tools": plan.tools, "rationale": plan.rationale},
        })

        requires_tools = plan.mode in {"calculator", "hybrid"} and bool(plan.tools)

        # --- INPUT RESOLUTION ---
        yield ("machine", {"node": "inputs", "status": "active", "title": "Input Resolution", "detail": "Extracting values from your query..."})
        extraction = extract_inputs(
            query=query,
            planned_tools=plan.tools,
            tool_registry=self.tool_registry,
            llm=self.orchestrator_llm,
            settings=self.settings,
        )
        user_inputs = extraction.user_inputs
        assumed_inputs = extraction.assumed_inputs
        assumptions = extraction.assumptions
        yield ("machine", {
            "node": "inputs", "status": "done", "title": "Input Resolution",
            "detail": f"Found {len(user_inputs)} values, filled {len(assumed_inputs)} defaults.",
            "meta": {"user_inputs": user_inputs, "assumed_inputs": assumed_inputs, "assumptions": assumptions},
        })

        # --- RETRIEVAL ---
        yield ("machine", {"node": "retrieval", "status": "active", "title": "Database Search", "detail": "Searching EC3 clauses..."})
        retrieved: list[RetrievedClause] = []
        retrieval_trace: list[dict[str, object]] = []
        for retrieval_event in self.retriever.iter_retrieve(query, top_k=self.settings.top_k_clauses):
            etype = retrieval_event.get("type")
            if etype == "iteration":
                step = retrieval_event.get("step", {})
                top = retrieval_event.get("top", [])
                top_labels = ", ".join(str(i.get("clause_id", "?")) for i in top) or "none"
                yield ("machine", {
                    "node": "retrieval", "status": "active", "title": "Database Search",
                    "detail": f"Pass {step.get('iteration', '?')}: found {len(step.get('top_clause_ids', []))} matches",
                    "meta": {"iteration": step, "top": top},
                })
            elif etype == "recursive":
                yield ("machine", {"node": "retrieval", "status": "active", "title": "Database Search", "detail": str(retrieval_event.get("detail", "Expanding search..."))})
            elif etype == "final":
                retrieved = retrieval_event.get("results", [])
                retrieval_trace = retrieval_event.get("trace", [])

        yield ("machine", {
            "node": "retrieval", "status": "done", "title": "Database Search",
            "detail": f"Retrieved {len(retrieved)} relevant clauses.",
            "meta": {
                "retrieved_count": len(retrieved),
                "top_clauses": [
                    {"doc_id": i.clause.doc_id, "clause_id": i.clause.clause_id, "title": i.clause.clause_title, "pointer": i.clause.pointer}
                    for i in retrieved[:5]
                ],
            },
        })

        # --- TOOLS ---
        tool_trace: list[ToolTraceStep] = []
        tool_outputs: dict[str, dict[str, Any]] = {}

        if not plan.tools:
            yield ("machine", {"node": "tools", "status": "done", "title": "MCP Tools", "detail": "No tools needed — retrieval-only path."})
        else:
            yield ("machine", {"node": "tools", "status": "active", "title": "MCP Tools", "detail": f"Running {len(plan.tools)} tool(s)..."})
            for idx, tool_name in enumerate(plan.tools, 1):
                inputs = self._build_tool_inputs(tool_name, extraction, tool_outputs)
                yield ("machine", {
                    "node": "tools", "status": "active", "title": "MCP Tools",
                    "detail": f"[{idx}/{len(plan.tools)}] {tool_name}",
                    "meta": {"tool": tool_name, "inputs": inputs},
                })
                try:
                    payload = self.tool_runner.run(tool_name, inputs)
                    tool_outputs[tool_name] = payload.get("result", {})
                    tool_trace.append(ToolTraceStep(tool_name=tool_name, status="ok", inputs=inputs, outputs=payload.get("result", {})))
                    yield ("machine", {"node": "tools", "status": "active", "title": "MCP Tools", "detail": f"{tool_name} — done", "meta": {"tool": tool_name, "status": "ok"}})
                except Exception as exc:
                    tool_trace.append(ToolTraceStep(tool_name=tool_name, status="error", inputs=inputs, error=str(exc)))
                    yield ("machine", {"node": "tools", "status": "error", "title": "MCP Tools", "detail": f"{tool_name} failed: {exc}", "meta": {"tool": tool_name, "status": "error"}})

            ts = "error" if any(s.status == "error" for s in tool_trace) else "done"
            yield ("machine", {"node": "tools", "status": ts, "title": "MCP Tools", "detail": f"Tools finished ({ts})."})

        # --- COMPOSE ---
        yield ("machine", {"node": "compose", "status": "active", "title": "Composing Answer", "detail": "Building grounded response..."})

        sources = self._collect_sources(retrieved, tool_outputs)
        supported = bool(sources)
        if requires_tools and any(s.status == "error" for s in tool_trace):
            supported = False

        narrative = self._draft_grounded_narrative(query=query, plan=plan, retrieved=retrieved, tool_outputs=tool_outputs, supported=supported)
        answer = self._build_markdown_answer(
            query=query, plan=plan, narrative=narrative, supported=supported,
            user_inputs=user_inputs, assumed_inputs=assumed_inputs, assumptions=assumptions,
            retrieved=retrieved, tool_outputs=tool_outputs, sources=sources, tool_trace=tool_trace,
        )

        yield ("machine", {
            "node": "compose", "status": "done" if supported else "error", "title": "Composing Answer",
            "detail": "Response ready." if supported else "Limited source support.",
            "meta": {
                "supported": supported,
                "used_tools": [s.tool_name for s in tool_trace if s.status == "ok"],
                "used_sources": [{"doc_id": s.doc_id, "clause_id": s.clause_id, "pointer": s.pointer} for s in sources[:8]],
            },
        })

        # --- OUTPUT ---
        yield ("machine", {"node": "output", "status": "active", "title": "Streaming", "detail": "Sending response..."})

        what_i_used = self._build_what_i_used(plan, retrieval_trace, tool_trace)
        response = ChatResponse(
            answer=answer, supported=supported,
            user_inputs=user_inputs, assumed_inputs=assumed_inputs, assumptions=assumptions,
            sources=sources, tool_trace=tool_trace,
            retrieval_trace=[RetrievalTraceStep.model_validate(s) for s in retrieval_trace],
            what_i_used=what_i_used,
        )
        yield ("machine", {"node": "output", "status": "done", "title": "Streaming", "detail": "Complete."})
        yield ("response", response)

    # ---- PLANNING ----
    def _build_plan(self, query: str) -> PlanResult:
        valid_tools = list(self.tool_registry.keys())

        if not self.orchestrator_llm.available:
            return PlanResult(
                mode="retrieval_only", tools=[],
                rationale="LLM unavailable — retrieval-only fallback.",
            )

        tool_descriptions = "\n".join(
            f"- {name}: {self.tool_registry[name].description} "
            f"| inputs: {list(self.tool_registry[name].input_schema.get('properties', {}).keys())}"
            for name in valid_tools
        )

        doc_descriptions = "\n".join(
            f"- {entry.standard} ({entry.year_version}): {entry.title}"
            for entry in self.document_registry
        )
        if not doc_descriptions:
            doc_descriptions = "(no documents loaded)"

        try:
            raw = self.orchestrator_llm.generate(
                system_prompt=(
                    "You are the Central Intelligence Orchestrator for a Eurocodes engineering chatbot.\n"
                    "Plan the best execution path for answering a user query.\n"
                    "You have access to:\n"
                    "1. A database of Eurocode clauses (for explanations, rules, formulas)\n"
                    "2. Calculator tools (for numerical computations)\n\n"
                    "Return JSON only: {\"mode\":\"retrieval_only|calculator|hybrid\","
                    "\"tools\":[...],\"rationale\":\"...\"}"
                ),
                user_prompt=(
                    "###TASK:PLAN###\n"
                    f"User query: {query}\n\n"
                    f"Available calculator tools:\n{tool_descriptions}\n\n"
                    f"Available Eurocode documents in the database:\n{doc_descriptions}\n\n"
                    "Decision rules:\n"
                    "- 'retrieval_only': conceptual/explanatory questions, clause lookups, "
                    "'what is'/'explain'/'describe' type queries\n"
                    "- 'calculator': pure computation — user gives values and wants numerical results\n"
                    "- 'hybrid': needs both clause evidence AND computation "
                    "(most common for engineering design checks)\n"
                    "- Order tools in execution dependency order "
                    "(e.g. section_classification before member_resistance)\n"
                    "- Only select tools that are directly relevant to the query\n"
                    "- If unsure whether tools are needed, prefer 'hybrid' over 'calculator'"
                ),
                temperature=0,
                max_tokens=900,
            )
            parsed = parse_json_loose(raw)
            if not isinstance(parsed, dict):
                raise ValueError("plan payload is not a JSON object")
            mode = parsed.get("mode", "retrieval_only")
            tools = [t for t in parsed.get("tools", []) if t in valid_tools]
            rationale = str(parsed.get("rationale", "LLM-generated plan."))
            if mode in {"retrieval_only", "calculator", "hybrid"}:
                if mode == "retrieval_only" and not tools:
                    heuristic = self._heuristic_plan_fallback(
                        query=query, valid_tools=valid_tools
                    )
                    if heuristic.tools:
                        return heuristic
                return PlanResult(mode=mode, tools=tools, rationale=rationale)
        except Exception as exc:
            logger.warning("plan_generation_failed", extra={"error": str(exc)})

        return self._heuristic_plan_fallback(query=query, valid_tools=valid_tools)

    def _heuristic_plan_fallback(self, *, query: str, valid_tools: list[str]) -> PlanResult:
        lowered = query.lower()
        has_section_profile = bool(
            re.search(r"\b(?:ipe|hea|heb|hem)\s*\d{2,4}\b", lowered)
        )
        has_calc_intent = any(
            token in lowered
            for token in (
                "resistance",
                "resistan",
                "capacity",
                "m_rd",
                "n_rd",
                "v_rd",
                "calculate",
                "calculation",
                "check",
                "design value",
            )
        )
        has_action_type = any(token in lowered for token in ("shear", "bending", "axial"))

        def pick(candidates: list[str]) -> list[str]:
            return [name for name in candidates if name in valid_tools]

        if any(token in lowered for token in ("interaction", "combined")):
            tools = pick(["section_classification_ec3", "member_resistance_ec3", "interaction_check_ec3"])
            if tools:
                return PlanResult(
                    mode="hybrid",
                    tools=tools,
                    rationale="Heuristic tool selection for interaction check query.",
                )

        if has_calc_intent and (has_section_profile or has_action_type):
            tools = pick(["section_classification_ec3", "member_resistance_ec3"])
            if tools:
                return PlanResult(
                    mode="hybrid",
                    tools=tools,
                    rationale="Heuristic tool selection for member resistance query.",
                )

        if any(token in lowered for token in ("bolt", "m12", "m16", "m20", "m24")) and "shear" in lowered:
            tools = pick(["bolt_shear_ec3"])
            if tools:
                return PlanResult(
                    mode="hybrid",
                    tools=tools,
                    rationale="Heuristic tool selection for bolt shear query.",
                )

        if "column buckling" in lowered or "flexural buckling" in lowered:
            tools = pick(["column_buckling_ec3"])
            if tools:
                return PlanResult(
                    mode="hybrid",
                    tools=tools,
                    rationale="Heuristic tool selection for column buckling query.",
                )

        return PlanResult(
            mode="retrieval_only",
            tools=[],
            rationale="LLM plan parsing failed and no heuristic tool match.",
        )

    # ---- FOLLOW-UP RESOLUTION ----
    def _resolve_followup(self, query: str, history: list) -> str:
        """Expand short follow-up queries using conversation history."""
        if not history:
            return query

        lowered = query.lower().strip()
        is_short = len(query.split()) <= 10
        followup_phrases = [
            "same but", "do it", "again", "now for", "now with",
            "repeat", "what about", "how about", "and for",
            "change to", "try with", "instead of", "but for",
            "ok ", "ok,", "for s", "with s", "use s",
        ]
        is_referential = any(phrase in lowered for phrase in followup_phrases)

        if not is_short and not is_referential:
            return query

        anchor_msg = None
        for msg in history:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if role == "user" and content:
                anchor_msg = content
                break

        if not anchor_msg:
            return query

        followup_values = {
            m.lower().replace(" ", "") for m in _FOLLOWUP_VALUE_RE.findall(query)
        }

        if self.orchestrator_llm.available:
            try:
                raw = self.orchestrator_llm.generate(
                    system_prompt=(
                        "You expand short follow-up messages into self-contained engineering queries. "
                        "Keep the ORIGINAL intent and all technical parameters from the first question. "
                        "Override only what the follow-up explicitly changes. "
                        "Return ONLY the expanded query as a single sentence. No explanation."
                    ),
                    user_prompt=(
                        f"Original question: {anchor_msg}\n"
                        f"Follow-up: {query}\n\n"
                        "Expanded self-contained query:"
                    ),
                    temperature=0,
                    max_tokens=150,
                )
                resolved = raw.strip()
                if resolved and len(resolved) > len(query):
                    if not followup_values or all(
                        v in resolved.lower().replace(" ", "")
                        for v in followup_values
                    ):
                        logger.info("followup_resolved", extra={"original": query, "resolved": resolved})
                        return resolved
                    logger.info("followup_llm_drift", extra={"resolved": resolved})
            except Exception as exc:
                logger.warning("followup_resolution_failed", extra={"error": str(exc)})

        logger.info("followup_heuristic", extra={"original": query, "context": anchor_msg[:120]})
        return f"{query}. {anchor_msg}"

    # ---- TOOL INPUT BUILDERS ----
    def _build_tool_inputs(
        self,
        tool_name: str,
        extraction: ExtractionResult,
        tool_outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        base = dict(extraction.tool_inputs.get(tool_name, {}))

        if not base:
            base = {**extraction.assumed_inputs, **extraction.user_inputs}

        for input_key, (src_tool, output_path, default) in _CHAIN_MAPPINGS.get(tool_name, {}).items():
            if base.get(input_key) is None:
                val = _resolve_nested(tool_outputs.get(src_tool, {}), output_path)
                base[input_key] = val if val is not None else default

        return {k: v for k, v in base.items() if v is not None}

    # ---- SOURCE COLLECTION ----
    def _collect_sources(self, retrieved: list[RetrievedClause], tool_outputs: dict[str, dict[str, Any]]) -> list[Citation]:
        seen: set[str] = set()
        sources: list[Citation] = []
        for payload in tool_outputs.values():
            for ref in payload.get("clause_references", []):
                doc_id = str(ref.get("doc_id", "unknown-doc"))
                clause_id = str(ref.get("clause_id", "unknown-clause"))
                title = str(ref.get("title", "Tool-linked clause"))
                pointer = str(ref.get("pointer", "tool-output"))
                address = build_citation_address(doc_id, clause_id, pointer)
                if address in seen:
                    continue
                seen.add(address)
                sources.append(Citation(doc_id=doc_id, clause_id=clause_id, clause_title=title, pointer=pointer, citation_address=address))
        for item in retrieved:
            c = Citation(doc_id=item.clause.doc_id, clause_id=item.clause.clause_id, clause_title=item.clause.clause_title, pointer=item.clause.pointer, citation_address=item.clause.citation_address)
            if c.citation_address not in seen:
                seen.add(c.citation_address)
                sources.append(c)
        return sources

    # ---- NARRATIVE GENERATION ----
    def _draft_grounded_narrative(self, *, query: str, plan: PlanResult, retrieved: list[RetrievedClause], tool_outputs: dict[str, dict[str, Any]], supported: bool) -> str:
        if not supported:
            return "I don't have enough information in the currently indexed clauses or tools to give you a reliable answer on this. You'd need to add the relevant Eurocode section or a dedicated calculator tool."

        if not self.orchestrator_llm.available:
            return self._build_fallback_narrative(query, plan, retrieved, tool_outputs)

        clause_evidence = []
        for c in retrieved[:5]:
            snippet = c.clause.text.strip()[:300]
            clause_evidence.append(f"[{c.clause.clause_id} — {c.clause.clause_title}]: {snippet}")

        tool_evidence: dict[str, Any] = {}
        for tname, tout in tool_outputs.items():
            tool_evidence[tname] = {
                "inputs": tout.get("inputs_used", {}),
                "outputs": tout.get("outputs", {}),
                "notes": tout.get("notes", []),
            }

        try:
            raw = self.orchestrator_llm.generate(
                system_prompt=(
                    "You are a senior structural engineer giving a concise answer to a colleague. "
                    "Rules:\n"
                    "1. First sentence: state the key result with its value and units. Example: 'The design bending resistance is **M_Rd = 223.08 kNm**'.\n"
                    "2. Then 1-2 sentences on the method/formula used. Mention the governing clause once, e.g. (EC3-1-1, Cl. 6.2.5).\n"
                    "3. Bold **key numerical results** with their engineering symbols.\n"
                    "4. Use ONLY the provided evidence. Never invent values.\n"
                    "5. Keep it to 2-4 sentences total. No sections, no bullet lists, no 'Sources' or 'Assumptions'.\n"
                    "6. Write naturally, as if explaining at a desk review. Always finish your sentences.\n"
                    "7. Ensure markdown emphasis is balanced (never leave dangling '**')."
                ),
                user_prompt=(
                    f"Question: {query}\n\n"
                    f"Retrieved EC3 clauses:\n" + "\n".join(clause_evidence) + "\n\n"
                    f"Tool results:\n{json.dumps(tool_evidence, default=str)}\n\n"
                    "Write a concise answer starting with the result."
                ),
                temperature=0.15,
                max_tokens=700,
            )
            return self._polish_narrative(
                raw.strip(),
                query=query,
                plan=plan,
                retrieved=retrieved,
                tool_outputs=tool_outputs,
            )
        except Exception as exc:
            logger.warning("answer_generation_failed", extra={"error": str(exc)})
            return self._build_fallback_narrative(query, plan, retrieved, tool_outputs)

    def _build_fallback_narrative(
        self,
        query: str,
        plan: PlanResult,
        retrieved: list[RetrievedClause],
        tool_outputs: dict[str, dict[str, Any]],
    ) -> str:
        parts: list[str] = []
        headline = self._compose_result_headline(
            plan=plan, tool_outputs=tool_outputs, query=query
        )
        if headline:
            parts.append(headline)

        if tool_outputs:
            first_payload = next(iter(tool_outputs.values()))
            outputs = first_payload.get("outputs", {})
            formula = outputs.get("formula")
            if isinstance(formula, str) and formula.strip():
                parts.append(f"I used the calculator output formula `{formula.strip()}` with the extracted section/material inputs.")

        basis = self._compose_clause_basis_sentence(retrieved=retrieved, tool_outputs=tool_outputs)
        if basis:
            parts.append(basis)

        text = " ".join(parts).strip()
        if not text:
            text = "Results computed from the available tools and EC3 database."
        return self._polish_narrative(
            text,
            query=query,
            plan=plan,
            retrieved=retrieved,
            tool_outputs=tool_outputs,
        )

    def _polish_narrative(
        self,
        narrative: str,
        *,
        query: str,
        plan: PlanResult,
        retrieved: list[RetrievedClause],
        tool_outputs: dict[str, dict[str, Any]],
    ) -> str:
        text = re.sub(r"\s+", " ", (narrative or "")).strip()
        if not text:
            text = "Results computed from the available tools and EC3 database."

        text = self._ensure_sentence_complete(text)
        text = self._sanitize_bold_markers(text)

        headline = self._compose_result_headline(
            plan=plan,
            tool_outputs=tool_outputs,
            query=query,
        )
        if headline and not self._has_unitized_headline(text):
            replaced = self._replace_leading_result_sentence(text, headline)
            text = replaced if replaced else f"{headline} {text}".strip()

        basis = self._compose_clause_basis_sentence(retrieved=retrieved, tool_outputs=tool_outputs)
        if basis and "Cl." not in text:
            text = f"{text} {basis}".strip()

        return self._ensure_sentence_complete(text)

    @staticmethod
    def _sanitize_bold_markers(text: str) -> str:
        return text.replace("**", "") if text.count("**") % 2 else text

    @staticmethod
    def _ensure_sentence_complete(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return cleaned

        terminal = cleaned[-1] if cleaned[-1] in ".!?" else ""
        body = cleaned[:-1].rstrip() if terminal else cleaned

        if _BROKEN_HYPHEN_TAIL_RE.search(body):
            body = _BROKEN_HYPHEN_TAIL_RE.sub("", body).rstrip(" ,;:-")
            cause_match = re.search(r"\b(?:because|due to|as)\b", body, re.IGNORECASE)
            if cause_match:
                prefix = body[: cause_match.start()].rstrip(" ,;:-")
                if prefix:
                    body = f"{prefix} because required section properties are missing"
                else:
                    body = "Required section properties are missing"
            elif body:
                body = f"{body} required section properties are missing"
            else:
                body = "Required section properties are missing"
            terminal = ""

        if _INCOMPLETE_CAUSE_TAIL_RE.search(body):
            cause_match = re.search(r"\b(?:because|due to|as)\b", body, re.IGNORECASE)
            if cause_match:
                prefix = body[: cause_match.start()].rstrip(" ,;:-")
                if prefix:
                    body = f"{prefix} because required section properties are missing"
                else:
                    body = "Required section properties are missing"
            else:
                body = f"{body} required section properties are missing"
            terminal = ""

        if _DANGLING_TAIL_RE.search(body):
            body = f"{body} additional required inputs are missing"
            terminal = ""

        cleaned = f"{body}{terminal}".strip()
        if cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        return cleaned

    @staticmethod
    def _has_unitized_headline(text: str) -> bool:
        first_sentence = text.split(".", 1)[0]
        has_equation = "=" in first_sentence
        has_unit = bool(re.search(r"\b(kNm|kN|MPa|GPa|mm|cm²|cm³|cm⁴|m)\b", first_sentence))
        return has_equation and has_unit

    def _replace_leading_result_sentence(self, text: str, headline: str) -> str:
        match = re.match(r"^(?P<first>.*?[.!?])(?:\s+(?P<rest>.*))?$", text.strip())
        if not match:
            return ""
        first = (match.group("first") or "").strip()
        rest = (match.group("rest") or "").strip()
        if not self._looks_like_result_sentence(first):
            return ""
        return f"{headline} {rest}".strip() if rest else headline

    @staticmethod
    def _looks_like_result_sentence(sentence: str) -> bool:
        cleaned = re.sub(r"<[^>]+>", "", sentence or "")
        cleaned = cleaned.replace("**", "").lower()
        if "=" not in cleaned:
            return False
        symbol_hit = any(token in cleaned for token in ("mrd", "m_rd", "nrd", "n_rd", "vrd", "v_rd"))
        result_word_hit = any(token in cleaned for token in ("resistance", "capacity", "utilization", "result"))
        return symbol_hit or result_word_hit

    def _compose_result_headline(
        self,
        *,
        plan: PlanResult,
        tool_outputs: dict[str, dict[str, Any]],
        query: str,
    ) -> str:
        _, payload = self._pick_tool_payload(plan=plan, tool_outputs=tool_outputs)
        if not payload:
            return ""

        outputs = payload.get("outputs", {})
        key, value = self._pick_primary_numeric_output(outputs, query=query)
        if not key:
            return ""

        base_key = self._strip_unit_suffix(key)
        symbol = self._pretty_key(base_key)
        value_text = self._format_value(key, value)
        quantity = self._describe_output_quantity(base_key)

        inputs = payload.get("inputs_used", {})
        section = str(inputs.get("section_name", "")).strip()
        steel = str(inputs.get("steel_grade", "")).strip()

        if section and steel:
            context = f"For **{section}** in **{steel}** steel, "
        elif section:
            context = f"For **{section}**, "
        elif steel:
            context = f"For **{steel}** steel, "
        else:
            context = ""

        return f"{context}the {quantity} is **{symbol} = {value_text}**."

    def _compose_clause_basis_sentence(
        self,
        *,
        retrieved: list[RetrievedClause],
        tool_outputs: dict[str, dict[str, Any]],
    ) -> str:
        seen: set[str] = set()
        clause_refs: list[tuple[str, str]] = []

        for payload in tool_outputs.values():
            for ref in payload.get("clause_references", []):
                cid = str(ref.get("clause_id", "")).strip()
                title = str(ref.get("title", "")).strip()
                norm = self._normalize_clause_id(cid)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                clause_refs.append((cid, title))
                if len(clause_refs) >= 2:
                    break
            if len(clause_refs) >= 2:
                break

        if len(clause_refs) < 2:
            for item in retrieved:
                cid = str(item.clause.clause_id).strip()
                title = str(item.clause.clause_title).strip()
                norm = self._normalize_clause_id(cid)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                clause_refs.append((cid, title))
                if len(clause_refs) >= 2:
                    break

        if not clause_refs:
            return ""
        if len(clause_refs) == 1:
            cid, title = clause_refs[0]
            return f"This is checked against EN 1993-1-1, Cl. {cid} ({title})."

        first_cid, first_title = clause_refs[0]
        second_cid, second_title = clause_refs[1]
        return (
            f"This follows EN 1993-1-1, Cl. {first_cid} ({first_title}) "
            f"with supporting classification guidance from Cl. {second_cid} ({second_title})."
        )

    def _pick_tool_payload(
        self,
        *,
        plan: PlanResult,
        tool_outputs: dict[str, dict[str, Any]],
    ) -> tuple[str, dict[str, Any]] | tuple[str, None]:
        for tool_name in plan.tools:
            payload = tool_outputs.get(tool_name)
            outputs = payload.get("outputs", {}) if payload else {}
            if payload and outputs and self._has_preferred_numeric_output(outputs):
                return tool_name, payload
        for tool_name, payload in tool_outputs.items():
            outputs = payload.get("outputs", {}) if payload else {}
            if payload and outputs and self._has_preferred_numeric_output(outputs):
                return tool_name, payload

        for tool_name in plan.tools:
            payload = tool_outputs.get(tool_name)
            if payload and payload.get("outputs"):
                return tool_name, payload
        for tool_name, payload in tool_outputs.items():
            if payload and payload.get("outputs"):
                return tool_name, payload
        return "", None

    @staticmethod
    def _has_preferred_numeric_output(outputs: dict[str, Any]) -> bool:
        preferred = {"M_Rd_kNm", "N_Rd_kN", "V_Rd_kN", "MEd_kNm", "NEd_kN", "utilization"}
        for key in preferred:
            value = outputs.get(key)
            if isinstance(value, (int, float)):
                return True
        return False

    @staticmethod
    def _pick_primary_numeric_output(
        outputs: dict[str, Any],
        *,
        query: str,
    ) -> tuple[str, float | int] | tuple[str, None]:
        if not outputs:
            return "", None

        lowered_query = query.lower()
        if any(token in lowered_query for token in ("shear", "v_rd", "shear resistance")):
            preferred_order = ["V_Rd_kN", "M_Rd_kNm", "N_Rd_kN", "utilization", "MEd_kNm", "NEd_kN"]
        elif any(token in lowered_query for token in ("axial", "compression", "tension", "n_rd", "normal force")):
            preferred_order = ["N_Rd_kN", "M_Rd_kNm", "V_Rd_kN", "utilization", "MEd_kNm", "NEd_kN"]
        elif any(token in lowered_query for token in ("bending", "moment", "m_rd")):
            preferred_order = ["M_Rd_kNm", "V_Rd_kN", "N_Rd_kN", "utilization", "MEd_kNm", "NEd_kN"]
        else:
            preferred_order = ["M_Rd_kNm", "N_Rd_kN", "V_Rd_kN", "MEd_kNm", "NEd_kN", "utilization"]

        for key in preferred_order:
            value = outputs.get(key)
            if isinstance(value, (int, float)):
                return key, value

        for key, value in outputs.items():
            if isinstance(value, (int, float)):
                return key, value
        return "", None

    @staticmethod
    def _strip_unit_suffix(key: str) -> str:
        for suffix, _ in _UNIT_SUFFIXES:
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return key

    @staticmethod
    def _describe_output_quantity(base_key: str) -> str:
        lowered = base_key.lower()
        if "m_rd" in lowered:
            return "design bending resistance"
        if "n_rd" in lowered:
            return "design axial resistance"
        if "v_rd" in lowered:
            return "design shear resistance"
        if "utilization" in lowered:
            return "utilization ratio"
        return "design result"

    @staticmethod
    def _pretty_tool_name(tool_name: str) -> str:
        pretty = tool_name.replace("_ec3", "").replace("_", " ").title()
        pretty = pretty.replace("Ipe", "IPE").replace("Ec3", "EC3")
        return pretty

    # ---- MARKDOWN BUILDER ----
    def _build_markdown_answer(self, *, query: str, plan: PlanResult, narrative: str, supported: bool,
                                user_inputs: dict[str, Any], assumed_inputs: dict[str, Any], assumptions: list[str],
                                retrieved: list[RetrievedClause], tool_outputs: dict[str, dict[str, Any]],
                                sources: list[Citation], tool_trace: list[ToolTraceStep]) -> str:
        lines: list[str] = []

        if narrative:
            lines.append(narrative)
        elif not supported:
            lines.append("I can't provide a grounded answer with the currently available sources and tools.")
        else:
            lines.append("Here are the results based on the EC3 database and calculation tools.")

        if tool_outputs:
            lines.append("")
            for tool_name in plan.tools:
                payload = tool_outputs.get(tool_name)
                if not payload:
                    continue
                outputs = payload.get("outputs", {})
                inputs_used = payload.get("inputs_used", {})
                if not outputs:
                    continue

                pretty = self._pretty_tool_name(tool_name)
                lines.append(f"<details class=\"tool-result\">")
                lines.append(f"<summary><strong>{pretty}</strong> detailed results</summary>")
                lines.append("")

                if inputs_used:
                    lines.append("**Inputs**")
                    lines.append("")
                    lines.append("| Input | Value |")
                    lines.append("| --- | --- |")
                    for k, v in inputs_used.items():
                        lines.append(f"| {self._pretty_key(k)} | {self._format_value(k, v)} |")
                    lines.append("")

                lines.append("**Outputs**")
                lines.append("")
                lines.append("| Output | Value |")
                lines.append("| --- | --- |")
                for k, v in outputs.items():
                    if isinstance(v, (int, float, str, bool)):
                        lines.append(f"| {self._pretty_key(k)} | **{self._format_value(k, v)}** |")
                lines.append("")

                notes = payload.get("notes", [])
                if notes:
                    lines.append("**Notes**")
                    lines.append("")
                    for n in notes:
                        lines.append(f"- {n}")
                    lines.append("")

                lines.append("</details>")
                lines.append("")

        if assumptions:
            lines.append("<details class=\"assumptions\">")
            lines.append("<summary>Assumptions used</summary>")
            lines.append("")
            for a in assumptions:
                lines.append(f"- {a}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if any(s.status == "error" for s in tool_trace):
            lines.append("\n**Tool Errors:**\n")
            for s in tool_trace:
                if s.status == "error":
                    lines.append(f"- {s.tool_name}: {s.error}")

        if sources:
            filtered = self._select_relevant_sources(
                narrative=narrative, sources=sources,
                retrieved=retrieved, tool_outputs=tool_outputs,
            )
            if not filtered:
                filtered = sources
            relevant = [s for s in filtered if s.clause_id and s.clause_id != "0" and s.clause_title and s.clause_title != "text"]
            if relevant:
                lines.append("\n---\n")
                lines.append("**References:**")
                seen_refs: set[str] = set()
                ref_idx = 0
                for s in relevant[:5]:
                    ref_key = self._normalize_clause_id(s.clause_id)
                    if ref_key in seen_refs:
                        continue
                    seen_refs.add(ref_key)
                    ref_idx += 1
                    lines.append(f"{ref_idx}. EN 1993-1-1, Cl. {s.clause_id} — {s.clause_title}")

        return self._format_subscripts("\n".join(lines).strip())

    def _format_value(self, key: str, val: Any) -> str:
        if isinstance(val, bool):
            return "PASS ✓" if val else "FAIL ✗"
        if isinstance(val, float):
            unit = self._guess_unit(key)
            if val == int(val) and abs(val) < 1e6:
                return f"{int(val)} {unit}".strip()
            return f"{val:.2f} {unit}".strip()
        if isinstance(val, int):
            unit = self._guess_unit(key)
            return f"{val} {unit}".strip()
        return str(val)

    def _pretty_key(self, key: str) -> str:
        unit = ""
        base = key
        for suffix, label in _UNIT_SUFFIXES:
            if base.endswith(suffix):
                unit = label
                base = base[: -len(suffix)]
                break

        sorted_subs = sorted(_KEY_SUBSCRIPTS.items(), key=lambda x: -len(x[0]))
        for pattern, replacement in sorted_subs:
            if base == pattern:
                return replacement + unit
            if base.startswith(pattern + "_"):
                rest = base[len(pattern) + 1 :].replace("_", " ")
                return f"{replacement} {rest}" + unit

        return base.replace("_", " ") + unit

    @staticmethod
    def _format_subscripts(text: str) -> str:
        for pattern, replacement in _NARRATIVE_SUB_RE:
            text = pattern.sub(replacement, text)
        return text

    def _guess_unit(self, key: str) -> str:
        key_lower = key.lower()
        if "knm" in key_lower: return "kNm"
        if "kn" in key_lower: return "kN"
        if "mpa" in key_lower: return "MPa"
        if "gpa" in key_lower: return "GPa"
        if "_mm" in key_lower: return "mm"
        if key_lower.endswith("_m"): return "m"
        if "cm2" in key_lower: return "cm²"
        if "cm3" in key_lower: return "cm³"
        if "cm4" in key_lower: return "cm⁴"
        return ""

    @staticmethod
    def _normalize_clause_id(clause_id: str) -> str:
        idx = clause_id.find("(")
        return clause_id[:idx] if idx > 0 else clause_id

    def _select_relevant_sources(
        self,
        *,
        narrative: str,
        sources: list[Citation],
        retrieved: list[RetrievedClause],
        tool_outputs: dict[str, dict[str, Any]],
    ) -> list[Citation]:
        inline_ids: set[str] = set()
        for match in re.finditer(r"Cl\.\s*([\d.]+)", narrative):
            inline_ids.add(match.group(1))

        tool_ids: set[str] = set()
        for payload in tool_outputs.values():
            for ref in payload.get("clause_references", []):
                cid = str(ref.get("clause_id", ""))
                if cid:
                    tool_ids.add(self._normalize_clause_id(cid))

        relevant_ids = inline_ids | tool_ids
        return [s for s in sources if self._normalize_clause_id(s.clause_id) in relevant_ids]

    def _build_what_i_used(self, plan: PlanResult, retrieval_trace: list[dict[str, object]], tool_trace: list[ToolTraceStep]) -> list[str]:
        summaries = [f"Plan: {plan.mode} — {plan.rationale}", f"Retrieval: {len(retrieval_trace)} search pass(es)"]
        if tool_trace:
            chain = " → ".join(s.tool_name for s in tool_trace)
            summaries.append(f"Tool chain: {chain}")
        return summaries
