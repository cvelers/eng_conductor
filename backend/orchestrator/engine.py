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
from backend.schemas import Attachment, ChatResponse, Citation, RetrievalTraceStep, ToolTraceStep
from backend.tools.response_formatter import ResponseFormatterTool
from backend.tools.runner import MCPToolRunner
from backend.utils.citations import build_citation_address
from backend.utils.json_utils import parse_json_loose
from backend.utils.parsing import ExtractionResult, extract_inputs

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    mode: str
    tools: list[str]
    rationale: str



def _flatten_tool_outputs(tool_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten all accumulated tool outputs into a single {param: value} dict."""
    flat: dict[str, Any] = {}
    for result in tool_outputs.values():
        for k, v in result.get("outputs", {}).items():
            flat[k] = v
    return flat


_FOLLOWUP_VALUE_RE = re.compile(
    r"\b(?:IPE\s*\d+|HEA\s*\d+|HEB\s*\d+|S(?:235|275|355|420|460)|M\d+|\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

_THINKING_MODES = {"standard", "thinking", "extended"}


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
        clauses: list[ClauseRecord] | None = None,
        response_formatter: ResponseFormatterTool | None = None,
    ) -> None:
        self.settings = settings
        self.orchestrator_llm = orchestrator_llm
        self.retriever = retriever
        self.tool_runner = tool_runner
        self.tool_registry = {entry.tool_name: entry for entry in tool_registry}
        self.document_registry = document_registry or []
        self._document_lookup: dict[str, DocumentRegistryEntry] = {
            entry.id: entry for entry in self.document_registry
        }
        self.clauses = clauses or []
        self._clause_lookup: dict[tuple[str, str], ClauseRecord] = {}
        for c in self.clauses:
            self._clause_lookup[(c.doc_id, c.clause_id)] = c
            norm = self._normalize_clause_id(c.clause_id)
            if norm and norm != c.clause_id:
                self._clause_lookup[(c.doc_id, norm)] = c
        self.response_formatter = response_formatter or ResponseFormatterTool()

    def run(
        self,
        query: str,
        *,
        history: list | None = None,
        thinking_mode: str = "thinking",
        attachments: list[Attachment] | None = None,
        is_edit: bool = False,
    ) -> ChatResponse:
        final_response: ChatResponse | None = None
        for event_type, payload in self.run_stream(
            query,
            history=history,
            thinking_mode=thinking_mode,
            attachments=attachments,
            is_edit=is_edit,
        ):
            if event_type == "response":
                final_response = payload
        if final_response is None:
            raise RuntimeError("Orchestrator did not produce a final response.")
        return final_response

    def run_stream(
        self,
        raw_query: str,
        *,
        history: list | None = None,
        thinking_mode: str = "thinking",
        attachments: list[Attachment] | None = None,
        is_edit: bool = False,
    ) -> Iterator[tuple[str, Any]]:
        attachments = attachments or []
        query = raw_query if is_edit else self._resolve_followup(raw_query, history or [])
        selected_mode = self._normalize_thinking_mode(thinking_mode)

        # --- INTENT CLASSIFICATION ---
        # Single multimodal classifier inspects text + images and decides what
        # pipeline stages are needed.
        classification = self._classify_intent(query, attachments)
        intent = classification["intent"]
        logger.info("intent_classified", extra={"intent": intent, "query_preview": query[:80]})

        if intent in ("decline", "greeting", "answer"):
            yield from self._handle_direct_response(query, attachments, intent)
            return

        # --- INTAKE ---
        yield ("machine", {"node": "intake", "status": "active", "title": "Query Intake", "detail": "Analyzing your question..."})
        plan = self._build_plan_for_mode(query, thinking_mode=selected_mode)
        yield ("machine", {"node": "intake", "status": "done", "title": "Query Intake", "detail": "Question understood."})
        yield ("machine", {
            "node": "plan", "status": "done", "title": "Pathway Planning",
            "detail": f"Strategy: {plan.mode} | Tools: {plan.tools or ['none']}",
            "meta": {
                "mode": plan.mode,
                "thinking_mode": selected_mode,
                "tools": plan.tools,
                "rationale": plan.rationale,
            },
        })

        run_retrieval = selected_mode in {"standard", "extended"} or plan.mode in {"retrieval_only", "hybrid"}
        run_tools = plan.mode in {"calculator", "hybrid"}
        retrieval_agentic: bool | None = None
        retrieval_recursive: bool | None = None
        if selected_mode == "standard":
            retrieval_agentic = False
            retrieval_recursive = False
        elif selected_mode == "extended":
            retrieval_agentic = True
            retrieval_recursive = True

        retrieved: list[RetrievedClause] = []
        retrieval_trace: list[dict[str, object]] = []

        # --- RETRIEVAL ---
        if run_retrieval:
            yield ("machine", {"node": "retrieval", "status": "active", "title": "Database Search", "detail": "Searching EC3 clauses..."})
            for retrieval_event in self.retriever.iter_retrieve(
                query,
                top_k=self.settings.top_k_clauses,
                agentic=retrieval_agentic,
                recursive=retrieval_recursive,
            ):
                etype = retrieval_event.get("type")
                if etype == "iteration":
                    step = retrieval_event.get("step", {})
                    top = retrieval_event.get("top", [])
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
                        for i in retrieved
                    ],
                },
            })
        else:
            yield ("machine", {
                "node": "retrieval", "status": "done", "title": "Database Search",
                "detail": "Skipped for calculator-only path.",
                "meta": {"retrieved_count": 0, "top_clauses": [], "skipped": True},
            })

        if selected_mode == "extended":
            yield ("machine", {
                "node": "plan",
                "status": "active",
                "title": "Pathway Planning",
                "detail": "Extended mode: selecting tools after retrieval...",
            })
            planned_tools, tool_plan_note = self._plan_tools_after_retrieval(
                query=query,
                retrieved=retrieved,
                initial_tools=plan.tools,
            )
            planned_tools = self._normalize_tool_chain(planned_tools)
            if planned_tools:
                rationale = plan.rationale
                if tool_plan_note:
                    rationale = f"{rationale} | {tool_plan_note}"
                plan = PlanResult(mode="hybrid", tools=planned_tools, rationale=rationale)
                run_tools = True
            else:
                rationale = f"{plan.rationale} | {tool_plan_note}" if tool_plan_note else plan.rationale
                plan = PlanResult(
                    mode="retrieval_only",
                    tools=[],
                    rationale=f"{rationale} | Extended mode found no required calculator.",
                )
                run_tools = False

            yield ("machine", {
                "node": "plan",
                "status": "done",
                "title": "Pathway Planning",
                "detail": f"Final strategy: {plan.mode} | Tools: {plan.tools or ['none']}",
                "meta": {
                    "mode": plan.mode,
                    "thinking_mode": selected_mode,
                    "tools": plan.tools,
                    "rationale": plan.rationale,
                },
            })
        elif run_tools and plan.mode == "hybrid":
            yield ("machine", {
                "node": "plan",
                "status": "active",
                "title": "Pathway Planning",
                "detail": "Refining tool plan from retrieved clauses...",
            })
            planned_tools, tool_plan_note = self._plan_tools_after_retrieval(
                query=query,
                retrieved=retrieved,
                initial_tools=plan.tools,
            )
            planned_tools = self._normalize_tool_chain(planned_tools)
            if planned_tools:
                rationale = plan.rationale
                if tool_plan_note:
                    rationale = f"{plan.rationale} | {tool_plan_note}"
                plan = PlanResult(mode="hybrid", tools=planned_tools, rationale=rationale)
            else:
                rationale = f"{plan.rationale} | {tool_plan_note}" if tool_plan_note else plan.rationale
                plan = PlanResult(
                    mode="retrieval_only",
                    tools=[],
                    rationale=f"{rationale} | Post-retrieval tool planning found no required calculator.",
                )
                run_tools = False

            yield ("machine", {
                "node": "plan",
                "status": "done",
                "title": "Pathway Planning",
                "detail": f"Final strategy: {plan.mode} | Tools: {plan.tools or ['none']}",
                "meta": {
                    "mode": plan.mode,
                    "thinking_mode": selected_mode,
                    "tools": plan.tools,
                    "rationale": plan.rationale,
                },
            })
        elif run_tools:
            normalized_tools = self._normalize_tool_chain(plan.tools)
            if normalized_tools != plan.tools:
                plan = PlanResult(
                    mode=plan.mode,
                    tools=normalized_tools,
                    rationale=f"{plan.rationale} | Tool chain normalized for execution dependencies.",
                )

        # --- INPUT RESOLUTION ---
        extraction = ExtractionResult(
            user_inputs={},
            assumed_inputs={},
            assumptions=[],
            tool_inputs={},
        )
        if run_tools:
            yield ("machine", {"node": "inputs", "status": "active", "title": "Input Resolution", "detail": "Extracting values from your query..."})
            extraction = extract_inputs(
                query=query,
                planned_tools=plan.tools,
                tool_registry=self.tool_registry,
                llm=self.orchestrator_llm,
                settings=self.settings,
            )
            yield ("machine", {
                "node": "inputs", "status": "done", "title": "Input Resolution",
                "detail": f"Found {len(extraction.user_inputs)} values, filled {len(extraction.assumed_inputs)} defaults.",
                "meta": {
                    "user_inputs": extraction.user_inputs,
                    "assumed_inputs": extraction.assumed_inputs,
                    "assumptions": extraction.assumptions,
                },
            })
        else:
            yield ("machine", {
                "node": "inputs", "status": "done", "title": "Input Resolution",
                "detail": "No calculator inputs required for this path.",
                "skipped": True,
                "meta": {"user_inputs": {}, "assumed_inputs": {}, "assumptions": []},
            })

        user_inputs = extraction.user_inputs
        assumed_inputs = extraction.assumed_inputs
        assumptions = extraction.assumptions

        requires_tools = run_tools

        # --- TOOLS ---
        tool_trace: list[ToolTraceStep] = []
        tool_outputs: dict[str, dict[str, Any]] = {}

        if not run_tools:
            yield ("machine", {"node": "tools", "status": "done", "title": "MCP Tools", "detail": "No tools needed for this strategy.", "skipped": True})
        elif not plan.tools:
            yield ("machine", {"node": "tools", "status": "error", "title": "MCP Tools", "detail": "Mode requires calculators, but no suitable tool chain was planned."})
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
        if requires_tools and (not plan.tools or any(s.status == "error" for s in tool_trace)):
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
                "used_sources": [{"doc_id": s.doc_id, "clause_id": s.clause_id, "pointer": s.pointer} for s in sources],
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
    def _normalize_thinking_mode(self, mode: str | None) -> str:
        normalized = str(mode or "thinking").strip().lower().replace("-", "_")
        if normalized in _THINKING_MODES:
            return normalized
        return "thinking"

    def _build_plan_for_mode(self, query: str, *, thinking_mode: str) -> PlanResult:
        if thinking_mode == "standard":
            return PlanResult(
                mode="retrieval_only",
                tools=[],
                rationale="Standard mode: database-only lookup with simple retrieval.",
            )

        plan = self._build_plan(query, thinking_mode=thinking_mode)
        if thinking_mode == "extended":
            forced_mode = "hybrid" if plan.tools else "retrieval_only"
            return PlanResult(
                mode=forced_mode,
                tools=plan.tools,
                rationale=f"{plan.rationale} | Extended mode enforces retrieval-first flow.",
            )
        return plan

    def _build_plan(self, query: str, *, thinking_mode: str = "thinking") -> PlanResult:
        valid_tools = list(self.tool_registry.keys())
        has_documents = bool(self.document_registry or self.clauses)

        registry_plan = self._registry_first_plan(
            query=query,
            valid_tools=valid_tools,
            has_documents=has_documents,
        )
        if registry_plan is not None:
            return registry_plan

        if not self.orchestrator_llm.available:
            return self._heuristic_plan_fallback(
                query=query,
                valid_tools=valid_tools,
                has_documents=has_documents,
            )

        tool_descriptions = "\n".join(
            f"- {name}: {self.tool_registry[name].description} "
            f"| tags: {self.tool_registry[name].tags} "
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
                    "You MUST reason over both registries before deciding:\n"
                    "1) Eurocode clause database registry\n"
                    "2) Calculator tool registry\n\n"
                    "Return JSON only: {\"mode\":\"retrieval_only|calculator|hybrid\","
                    "\"tools\":[...],\"rationale\":\"...\"}"
                ),
                user_prompt=(
                    "###TASK:PLAN###\n"
                    f"User query: {query}\n\n"
                    f"Thinking mode: {thinking_mode}\n\n"
                    f"Available calculator tools:\n{tool_descriptions}\n\n"
                    f"Available Eurocode documents in the database:\n{doc_descriptions}\n\n"
                    "Decision rules:\n"
                    "- 'retrieval_only': explanations, procedure/method checks, clause lookups, "
                    "or any query that does not require a numerical result\n"
                    "- 'calculator': direct/trivial numeric computation where clause retrieval is not needed\n"
                    "- 'hybrid': needs both clause evidence and computation, especially when procedure "
                    "must be verified before calculation\n"
                    "- If a direct numeric query is answerable by available tools and does NOT request clauses, prefer 'calculator'\n"
                    "- If the query explicitly asks for clause/citation/code basis, prefer retrieval_only or hybrid\n"
                    "- Order tools in execution dependency order "
                    "(e.g. section_classification before member_resistance)\n"
                    "- Only select tools that are directly relevant to the query\n"
                    "- Modes are equal choices: pick the single best mode for this query"
                ),
                temperature=0,
                max_tokens=4096,
            )
            parsed = parse_json_loose(raw)
            if not isinstance(parsed, dict):
                raise ValueError("plan payload is not a JSON object")
            mode = str(parsed.get("mode", "retrieval_only")).strip()
            tools = self._normalize_tool_chain(
                [t for t in parsed.get("tools", []) if t in valid_tools]
            )
            rationale = str(parsed.get("rationale", "LLM-generated plan."))
            if mode in {"retrieval_only", "calculator", "hybrid"}:
                if mode == "retrieval_only":
                    return PlanResult(mode="retrieval_only", tools=[], rationale=rationale)

                if not tools:
                    heuristic = self._heuristic_plan_fallback(
                        query=query,
                        valid_tools=valid_tools,
                        has_documents=has_documents,
                    )
                    tools = self._normalize_tool_chain(heuristic.tools)
                    if tools:
                        rationale = f"{rationale} | Tool chain recovered via heuristic fallback."
                return PlanResult(mode=mode, tools=tools, rationale=rationale)
        except Exception as exc:
            logger.warning("plan_generation_failed", extra={"error": str(exc)})

        return self._heuristic_plan_fallback(
            query=query,
            valid_tools=valid_tools,
            has_documents=has_documents,
        )

    def _registry_first_plan(
        self,
        *,
        query: str,
        valid_tools: list[str],
        has_documents: bool,
    ) -> PlanResult | None:
        if not valid_tools and has_documents:
            return PlanResult(
                mode="retrieval_only",
                tools=[],
                rationale="No calculator tools registered; using database-only path.",
            )
        if not valid_tools and not has_documents:
            return PlanResult(
                mode="retrieval_only",
                tools=[],
                rationale="No documents or tools are registered for this query.",
            )

        matched_tools = self._match_tools_for_query(query=query, valid_tools=valid_tools)
        intent = self._query_intent(query)

        if not has_documents:
            if matched_tools and intent.get("has_specific_values", True):
                return PlanResult(
                    mode="calculator",
                    tools=matched_tools,
                    rationale="No documents loaded; routed to matching calculator tools.",
                )
            return PlanResult(
                mode="retrieval_only",
                tools=[],
                rationale="No documents loaded and no specific input values provided.",
            )

        if intent["lookup_only"]:
            return PlanResult(
                mode="retrieval_only",
                tools=[],
                rationale="Detected clause/procedure lookup intent without required calculation.",
            )

        if matched_tools and intent["pure_calculation"]:
            return PlanResult(
                mode="calculator",
                tools=matched_tools,
                rationale="Pure calculation intent detected; retrieval skipped in thinking mode.",
            )

        if matched_tools and intent["has_calc_intent"] and intent["code_required"]:
            return PlanResult(
                mode="hybrid",
                tools=matched_tools,
                rationale="Query requires both code evidence and numeric calculation.",
            )

        return None

    def _query_intent(self, query: str) -> dict[str, bool]:
        lowered = query.lower()
        has_numeric = bool(re.search(r"\d", lowered))
        has_calc_intent = any(
            token in lowered
            for token in (
                "calculate",
                "calculation",
                "compute",
                "determine",
                # "what is" removed — question phrase, not a calculation request
                "max",
                "maximum",
                "deflection",
                "moment",
                "shear",
                "reaction",
                "resistance",
                "capacity",
                "utilization",
                "utilisation",
                "m_rd",
                "n_rd",
                "v_rd",
                "beam",
                "load",
                "span",
            )
        )
        has_lookup_intent = any(
            token in lowered
            for token in (
                "explain",
                "procedure",
                "method",
                "what does",
                "which clause",
                "clause",
                "citation",
                "reference",
                "requirement",
                "provision",
                "rule",
            )
        )
        code_required = bool(
            re.search(r"\b(?:en\s*1993|ec3|eurocode|cl\.)\b", lowered)
        ) or any(
            token in lowered
            for token in (
                "according to",
                "as per",
                "per ec3",
                "per en",
                "cite",
                "with clauses",
                "show clauses",
                "normative",
            )
        )

        # Detect specific engineering values (section names, grades, dimensions).
        # Without these, queries containing engineering terms are conceptual —
        # they should route to retrieval, not to calculator tools with defaults.
        has_specific_values = bool(re.search(
            r"\b(?:"
            r"(?:ipe|heb|hea|hem|ub|uc|chs|rhs|shs)\s*\d+"  # section: IPE300
            r"|s(?:235|275|355|420|460)\b"                     # steel: S355
            r"|m(?:12|14|16|20|22|24|27|30|36)\b"             # bolt: M20
            r"|(?:4\.6|4\.8|5\.6|5\.8|6\.8|8\.8|10\.9|12\.9)" # bolt grade: 8.8
            r"|\d+(?:\.\d+)?\s*(?:mm|cm|kn|mpa|n/mm|knm)"    # value+unit: 300mm
            r")",
            lowered,
        ))

        lookup_only = has_lookup_intent and not has_calc_intent and not has_numeric
        pure_calculation = (
            has_calc_intent
            and has_specific_values  # must have concrete inputs to fast-path
            and not code_required
            and not has_lookup_intent
        )
        if pure_calculation and any(token in lowered for token in ("explain", "why", "how")):
            pure_calculation = False

        return {
            "has_calc_intent": has_calc_intent,
            "has_lookup_intent": has_lookup_intent,
            "code_required": code_required,
            "lookup_only": lookup_only,
            "pure_calculation": pure_calculation,
            "has_specific_values": has_specific_values,
        }

    def _match_tools_for_query(self, *, query: str, valid_tools: list[str]) -> list[str]:
        lowered = query.lower()

        def pick(candidates: list[str]) -> list[str]:
            return self._normalize_tool_chain(
                [name for name in candidates if name in valid_tools]
            )

        if any(token in lowered for token in ("simply supported", "simple beam", "udl")):
            tools = pick(["simple_beam_calculator"])
            if tools:
                return tools

        if "cantilever" in lowered:
            tools = pick(["cantilever_beam_calculator"])
            if tools:
                return tools

        if any(token in lowered for token in ("interaction", "combined")) and any(
            token in lowered for token in ("bending", "axial", "moment", "compression", "tension")
        ):
            tools = pick(["section_classification_ec3", "member_resistance_ec3", "interaction_check_ec3"])
            if tools:
                return tools

        if any(token in lowered for token in ("bolt", "m12", "m16", "m20", "m24")) and "shear" in lowered:
            tools = pick(["bolt_shear_ec3"])
            if tools:
                return tools

        if "column buckling" in lowered or "flexural buckling" in lowered:
            tools = pick(["column_buckling_ec3"])
            if tools:
                return tools

        if "moment resistance" in lowered and "ipe" in lowered:
            tools = pick(["ipe_moment_resistance_ec3"])
            if tools:
                return tools

        if any(token in lowered for token in ("resistance", "capacity", "m_rd", "n_rd", "v_rd")):
            tools = pick(["section_classification_ec3", "member_resistance_ec3"])
            if tools:
                return tools

        query_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
        scored: list[tuple[int, str]] = []
        for name in valid_tools:
            entry = self.tool_registry.get(name)
            if entry is None:
                continue
            tool_tokens = set(
                re.findall(
                    r"[a-z0-9_]+",
                    f"{name} {entry.description} {' '.join(entry.tags)}".lower(),
                )
            )
            overlap = query_tokens & tool_tokens
            score = len(overlap)
            if score > 0:
                scored.append((score, name))

        scored.sort(key=lambda item: (-item[0], item[1]))
        if scored and scored[0][0] >= 2:
            return self._normalize_tool_chain([scored[0][1]])
        return []

    def _heuristic_plan_fallback(
        self,
        *,
        query: str,
        valid_tools: list[str],
        has_documents: bool,
    ) -> PlanResult:
        matched_tools = self._match_tools_for_query(query=query, valid_tools=valid_tools)
        intent = self._query_intent(query)

        if matched_tools:
            if has_documents and (intent["code_required"] or intent["has_lookup_intent"]):
                return PlanResult(
                    mode="hybrid",
                    tools=matched_tools,
                    rationale="Heuristic fallback selected hybrid path from query intent + tool match.",
                )
            if intent.get("has_specific_values", True):
                return PlanResult(
                    mode="calculator",
                    tools=matched_tools,
                    rationale="Heuristic fallback selected calculator-only path from query intent + tool match.",
                )
            # Tools matched but no specific input values — conceptual question
            return PlanResult(
                mode="retrieval_only",
                tools=[],
                rationale="Tools matched but no specific input values provided; using retrieval for conceptual answer.",
            )

        return PlanResult(
            mode="retrieval_only",
            tools=[],
            rationale="No tool match found; using retrieval-only fallback.",
        )

    def _normalize_tool_chain(
        self,
        tools: list[str],
        already_run: set[str] | None = None,
    ) -> list[str]:
        """Ensure prerequisite tools are included and properly ordered.

        Uses the LLM to determine if any prerequisite tools are missing
        (e.g. section_classification before member_resistance).  Falls
        back to exact-name schema matching when the LLM is unavailable.

        ``already_run`` — tools whose outputs are already available in the
        session context (from earlier tasks).  Only *implicitly added*
        prerequisites are skipped; tools explicitly requested in ``tools``
        always run (they may need different inputs in this task).
        """
        already_run = already_run or set()
        valid_tools = set(self.tool_registry.keys())
        planned = [t for t in tools if t in valid_tools]
        if not planned:
            return []
        explicitly_requested = set(planned)

        # --- LLM path: ask the model to resolve the complete chain ---
        if self.orchestrator_llm.available:
            resolved = self._llm_resolve_tool_chain(
                planned, already_run, explicitly_requested,
            )
            if resolved:
                return resolved

        # --- Fallback: exact-name output→input matching ---
        output_producers: dict[str, str] = {}
        for name, entry in self.tool_registry.items():
            out_props = (
                entry.output_schema
                .get("properties", {})
                .get("outputs", {})
                .get("properties", {})
            )
            for out_name in out_props:
                output_producers[out_name] = name

        all_needed = list(planned)
        planned_set = set(planned)
        for tool_name in list(planned):
            entry = self.tool_registry[tool_name]
            for in_name in entry.input_schema.get("properties", {}):
                if in_name in output_producers:
                    producer = output_producers[in_name]
                    if producer not in planned_set and producer in valid_tools:
                        all_needed.insert(0, producer)
                        planned_set.add(producer)

        # Deduplicate preserving order.
        # Skip already-run tools only if they were implicitly added as
        # prerequisites — explicitly requested tools always run.
        seen: set[str] = set()
        result: list[str] = []
        for t in all_needed:
            if t in seen:
                continue
            if t in already_run and t not in explicitly_requested:
                continue
            seen.add(t)
            result.append(t)
        return result

    # ---- LLM-BASED SESSION RESOLUTION ----

    def _llm_resolve_tool_chain(
        self,
        planned: list[str],
        already_run: set[str] | None = None,
        explicitly_requested: set[str] | None = None,
    ) -> list[str] | None:
        """Ask the LLM to verify / complete a tool chain with prerequisites."""
        already_run = already_run or set()
        explicitly_requested = explicitly_requested or set(planned)
        valid_tools = set(self.tool_registry.keys())
        tool_lines: list[str] = []
        for name, entry in self.tool_registry.items():
            in_params = list(entry.input_schema.get("properties", {}).keys())
            out_props = (
                entry.output_schema
                .get("properties", {})
                .get("outputs", {})
                .get("properties", {})
            )
            out_params = list(out_props.keys())
            tool_lines.append(
                f"- {name}: {entry.description}  "
                f"inputs={in_params}  outputs={out_params}"
            )

        # Only tell the LLM about prerequisite tools it can skip —
        # explicitly requested tools must always be included.
        skippable = already_run - explicitly_requested
        already_run_note = ""
        if skippable:
            already_run_note = (
                f"\nPrerequisite tools already executed (outputs available): "
                f"{json.dumps(sorted(skippable))}\n"
                "Do NOT include these as prerequisites — their results "
                "are already available and will be reused automatically.\n"
            )

        prompt = (
            "###TASK:RESOLVE_TOOL_CHAIN###\n"
            f"Selected tools: {json.dumps(planned)}\n\n"
            "All available tools:\n" + "\n".join(tool_lines) + "\n\n"
            + already_run_note +
            "Some tools produce outputs that another tool needs as input "
            "(names may differ, e.g. 'governing_class' maps to 'section_class').\n"
            "Return a JSON array of tool names in execution order, "
            "adding any missing prerequisite tools but excluding any "
            "already-executed prerequisite tools.\n"
            "Return ONLY the JSON array."
        )
        try:
            raw = self.orchestrator_llm.generate(
                system_prompt=(
                    "You determine tool execution order for engineering "
                    "calculations. Return only a valid JSON array."
                ),
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=2000,
            )
            parsed = parse_json_loose(raw)
            if isinstance(parsed, list) and parsed:
                return [
                    t for t in parsed
                    if t in valid_tools
                    and (t in explicitly_requested or t not in already_run)
                ]
        except Exception as exc:
            logger.warning("LLM tool-chain resolution failed: %s", exc)
        return None

    def _llm_resolve_inputs(
        self,
        tool_name: str,
        schema_props: dict[str, Any],
        params_to_resolve: list[str],
        current_inputs: dict[str, Any],
        all_tool_outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Use the LLM to resolve or verify tool inputs from session context.

        Called for parameters that are either truly unresolved (None) or
        that have decomposition-guessed values which may need overriding
        by actual computed results from prior tools.
        """
        context_lines: list[str] = []
        for src_tool, result in all_tool_outputs.items():
            outputs = result.get("outputs", {})
            if outputs:
                context_lines.append(f"  {src_tool}: {json.dumps(outputs)}")
        if not context_lines:
            return {}

        # Include current value (if any) so the LLM can decide to keep or override
        params_info: dict[str, Any] = {}
        for p in params_to_resolve:
            info: dict[str, Any] = {
                k: v
                for k, v in schema_props.get(p, {}).items()
                if k in ("type", "description", "minimum", "maximum")
            }
            cur = current_inputs.get(p)
            if cur is not None:
                info["current_value"] = cur
            params_info[p] = info

        prompt = (
            "###TASK:RESOLVE_INPUTS###\n"
            f"Tool: {tool_name}\n"
            f"Parameters to resolve: {json.dumps(params_info)}\n"
            f"Known inputs: {json.dumps({k: v for k, v in current_inputs.items() if v is not None})}\n\n"
            "Previous tool outputs:\n" + "\n".join(context_lines) + "\n\n"
            "For each parameter, determine the correct value from the prior "
            "tool outputs.  If a parameter has a current_value, override it "
            "if a prior tool computed a more accurate value (e.g. "
            "'governing_class' from section_classification maps to "
            "'section_class').  Return ONLY a JSON object of "
            "{param_name: value}.  Omit any that cannot be determined."
        )
        try:
            raw = self.orchestrator_llm.generate(
                system_prompt=(
                    "You resolve tool input parameters from available "
                    "session data. Return only valid JSON."
                ),
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=2000,
            )
            result = parse_json_loose(raw)
            if isinstance(result, dict):
                return result
        except Exception as exc:
            logger.warning(
                "LLM input resolution failed for %s: %s", tool_name, exc,
            )
        return {}

    def _plan_tools_after_retrieval(
        self,
        *,
        query: str,
        retrieved: list[RetrievedClause],
        initial_tools: list[str],
    ) -> tuple[list[str], str]:
        valid_tools = list(self.tool_registry.keys())
        fallback_tools = self._normalize_tool_chain(
            [tool for tool in initial_tools if tool in self.tool_registry]
        )

        if not self.orchestrator_llm.available:
            if fallback_tools:
                return fallback_tools, "Used initial tool plan (LLM unavailable for post-retrieval planning)."
            heuristic = self._heuristic_plan_fallback(
                query=query,
                valid_tools=valid_tools,
                has_documents=bool(self.document_registry or self.clauses),
            )
            return self._normalize_tool_chain(heuristic.tools), "Post-retrieval planner unavailable; used heuristic tools."

        tool_descriptions = "\n".join(
            f"- {name}: {self.tool_registry[name].description} "
            f"| inputs: {list(self.tool_registry[name].input_schema.get('properties', {}).keys())}"
            for name in valid_tools
        )
        retrieved_context = "\n".join(
            f"- {item.clause.standard}, Cl. {item.clause.clause_id}: {item.clause.clause_title}"
            for item in retrieved[:8]
        )
        if not retrieved_context:
            retrieved_context = "(no clauses retrieved)"

        try:
            raw = self.orchestrator_llm.generate(
                system_prompt=(
                    "You select calculator tools for a Eurocode 3 query AFTER retrieval has already run. "
                    "Return JSON only: {\"tools\":[...],\"rationale\":\"...\"}. "
                    "Pick the minimal necessary tools and order them by dependency."
                ),
                user_prompt=(
                    "###TASK:PLAN_TOOLS###\n"
                    f"User query: {query}\n\n"
                    f"Retrieved clause evidence:\n{retrieved_context}\n\n"
                    f"Available tools:\n{tool_descriptions}\n\n"
                    f"Initial tool proposal:\n{initial_tools}\n\n"
                    "Return JSON only."
                ),
                temperature=0,
                max_tokens=4096,
            )
            parsed = parse_json_loose(raw)
            if isinstance(parsed, dict):
                selected = self._normalize_tool_chain(
                    [tool for tool in parsed.get("tools", []) if tool in self.tool_registry]
                )
                rationale = str(parsed.get("rationale", "Tool plan refined from retrieval context."))
                if selected:
                    return selected, rationale
        except Exception as exc:
            logger.warning("post_retrieval_tool_plan_failed", extra={"error": str(exc)})

        if fallback_tools:
            return fallback_tools, "Kept initial tool plan after retrieval."

        heuristic = self._heuristic_plan_fallback(
            query=query,
            valid_tools=valid_tools,
            has_documents=bool(self.document_registry or self.clauses),
        )
        return self._normalize_tool_chain(heuristic.tools), "Used heuristic tool plan after retrieval."

    # ---- INTENT CLASSIFICATION ----
    #
    # Principled routing layer.  A single multimodal LLM call inspects BOTH the
    # text query AND any attached images, then returns a structured intent that
    # determines exactly which pipeline stages run.
    #
    # Intents
    # -------
    #   pipeline   – Full engineering pipeline (retrieval + optional tools).
    #   answer     – Engineering-related but can be answered conversationally
    #               by the LLM (e.g. "what's in this beam diagram?").  No
    #               database / tools needed.
    #   decline    – Clearly off-topic.  Polite decline, no LLM call.
    #   greeting   – Social pleasantry.  Short friendly reply.

    _ATTACHMENT_MARKER_RE = re.compile(
        r"\[Attached (?:image|file): [^\]]*\]\s*", re.IGNORECASE,
    )

    _ENG_KEYWORDS: list[str] = [
        "eurocode", "ec3", "ec2", "ec1", "ec0",
        "en 1993", "en 1992", "en 1991", "en 1990",
        "steel", "concrete", "beam", "column", "bending", "shear", "buckling",
        "section class", "resistance", "load", "uls", "sls", "moment",
        "axial", "bolt", "weld", "connection", "plate", "flange", "web",
        "elastic", "plastic", "yield", "mpa", "kn", "knm", "n/mm",
        "ipe", "hea", "heb", "chs", "rhs", "shs",
        "calculate", "check", "verify", "design",
        "deflection", "stiffness", "stability", "interaction",
        "partial factor", "gamma", "national annex", "clause",
        "reinforcement", "rebar", "prestress", "foundation", "footing",
        "truss", "frame", "bracing", "diaphragm", "cross-section",
        "structural", "civil engineer", "stress", "strain", "tension",
        "compression", "torsion", "fatigue", "seismic", "wind load",
        "snow load", "dead load", "live load", "imposed load",
    ]

    _GREETINGS = frozenset({
        "hi", "hello", "hey", "thanks", "thank you", "ok", "bye",
        "good morning", "good evening", "good afternoon", "sup",
        "yo", "howdy", "cheers",
    })

    # Static polite decline — no LLM call needed.
    _DECLINE_ANSWER = (
        "I'm the **EC3 Assistant** — a structural engineering chatbot specialised in "
        "**Eurocodes** (steel, concrete, timber design, structural calculations, etc.).\n\n"
        "This doesn't look like a structural engineering question, so I'm not the best "
        "fit here.  Feel free to ask me about things like:\n"
        "- Steel or concrete member design to Eurocodes\n"
        "- Section classification, resistance checks, buckling\n"
        "- Load combinations, ULS/SLS verifications\n"
        "- Connection design, bolt/weld checks\n\n"
        "How can I help you with structural engineering?"
    )

    _CLASSIFY_SYSTEM = (
        "You are a router for a structural / civil engineering chatbot that specialises "
        "in Eurocodes (EC0-EC9).  Given the user's text and any attached images, "
        "classify the request into EXACTLY one intent.\n\n"
        "Intents:\n"
        "  PIPELINE  – The user needs a calculation, code check, or detailed Eurocode "
        "lookup.  Requires the database and/or calculator tools.\n"
        "  ANSWER    – The query IS related to structural/civil engineering (or the "
        "attached image shows engineering content such as a structural drawing, beam "
        "diagram, steel section, construction plan, FEM model, Eurocode page, load "
        "diagram, etc.) BUT can be answered conversationally without a database search "
        "or calculator.  Examples: describing what is in an engineering image, explaining "
        "a structural concept, interpreting a drawing.\n"
        "  DECLINE   – The query and image(s) are clearly NOT related to structural "
        "engineering (e.g. food photos, selfies, animals, landscapes, general "
        "knowledge, coding questions, weather).\n"
        "  GREETING  – The message is a social pleasantry (hi, hello, thanks, bye).\n\n"
        "IMPORTANT: Look at the ACTUAL IMAGE CONTENT when images are attached.  "
        "A photo of a beam, column, structural drawing, building, or construction site "
        "is engineering content → ANSWER or PIPELINE, never DECLINE.\n\n"
        "Respond with ONLY the single word: PIPELINE, ANSWER, DECLINE, or GREETING."
    )

    def _classify_intent(
        self,
        query: str,
        attachments: list[Attachment],
    ) -> dict[str, str]:
        """Single entry-point for intent classification.

        Returns ``{"intent": "pipeline"|"answer"|"decline"|"greeting"}``.
        Uses heuristics first, falls back to a (multimodal) LLM call for
        ambiguous cases.
        """
        has_images = any(a.is_image and a.data_url for a in attachments)

        # Strip frontend attachment markers so we work on the real user text.
        cleaned = self._ATTACHMENT_MARKER_RE.sub("", query).strip()
        lowered = cleaned.lower()
        words = lowered.split()
        has_eng = any(kw in lowered for kw in self._ENG_KEYWORDS)

        # ---- Heuristic fast-paths (no LLM call) ----

        # H1: text explicitly contains engineering jargon AND asks for a
        #     calculation / check / design → full pipeline, no need to ask LLM.
        calc_verbs = {"calculate", "check", "verify", "design", "determine", "compute", "find"}
        if has_eng and any(v in lowered for v in calc_verbs):
            return {"intent": "pipeline"}

        # H2: greeting
        if len(words) <= 4 and lowered.rstrip("!.,?") in self._GREETINGS:
            return {"intent": "greeting"}

        # H3: text has engineering keywords but no calc verb and no images →
        #     could be a conceptual question; still send to pipeline to be safe
        #     (the compose step handles conversational answers too).
        if has_eng and not has_images:
            return {"intent": "pipeline"}

        # ---- LLM classification (multimodal when images present) ----
        llm_intent = self._llm_classify(cleaned, attachments, has_images)
        if llm_intent:
            return {"intent": llm_intent}

        # ---- Fallback when LLM is unavailable / fails ----
        if has_images:
            # We can't see the image without the LLM.  Since the text has no
            # engineering keywords, we don't know — route to 'answer' so the
            # LLM can at least TRY to respond with vision in the response step.
            return {"intent": "answer"}
        if len(words) <= 6:
            return {"intent": "decline"}
        return {"intent": "pipeline"}

    def _llm_classify(
        self,
        cleaned: str,
        attachments: list[Attachment],
        has_images: bool,
    ) -> str | None:
        """Call the orchestrator LLM to classify intent.

        Returns one of ``"pipeline"``, ``"answer"``, ``"decline"``,
        ``"greeting"`` — or *None* if the call fails.
        """
        if not self.orchestrator_llm.available:
            return None

        try:
            if has_images:
                # Build multimodal content: images first, then text prompt.
                content_parts: list[dict[str, Any]] = []
                for att in attachments:
                    if att.is_image and att.data_url:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": att.data_url},
                        })
                content_parts.append({
                    "type": "text",
                    "text": f"User message: \"{cleaned}\"" if cleaned else "User sent image(s) with no text.",
                })
                raw = self.orchestrator_llm.generate_multimodal(
                    system_prompt=self._CLASSIFY_SYSTEM,
                    content_parts=content_parts,
                    temperature=0,
                    max_tokens=256,
                    reasoning_effort="low",
                )
            else:
                raw = self.orchestrator_llm.generate(
                    system_prompt=self._CLASSIFY_SYSTEM,
                    user_prompt=f"User message: \"{cleaned}\"",
                    temperature=0,
                    max_tokens=256,
                    reasoning_effort="low",
                )

            token = raw.strip().upper().rstrip(".")
            logger.info("intent_classification_raw", extra={"raw": raw.strip(), "token": token, "has_images": has_images})

            mapping = {
                "PIPELINE": "pipeline",
                "ANSWER": "answer",
                "DECLINE": "decline",
                "GREETING": "greeting",
            }
            for key, val in mapping.items():
                if key in token:
                    return val

            # Model returned something unexpected — log it and fall through.
            logger.warning("intent_classification_unexpected", extra={"raw": raw.strip()})
            return None

        except Exception as exc:
            logger.warning("intent_classification_failed", extra={"error": str(exc)})
            return None

    # ---- DIRECT RESPONSE HANDLERS ----

    def _handle_direct_response(
        self,
        query: str,
        attachments: list[Attachment],
        intent: str,
    ) -> Iterator[tuple[str, Any]]:
        """Handle intents that bypass the full engineering pipeline.

        Supports three modes:
          greeting  – short friendly reply
          decline   – instant polite decline (no LLM call)
          answer    – conversational LLM answer (with vision if images present)
        """
        yield ("machine", {
            "node": "intake", "status": "active",
            "title": "Query Intake", "detail": "Analyzing your question...",
        })

        if intent == "decline":
            yield ("machine", {
                "node": "intake", "status": "done",
                "title": "Query Intake", "detail": "Off-topic query — quick response.",
            })
        elif intent == "greeting":
            yield ("machine", {
                "node": "intake", "status": "done",
                "title": "Query Intake", "detail": "Greeting received.",
            })
        else:
            yield ("machine", {
                "node": "intake", "status": "done",
                "title": "Query Intake", "detail": "Answering directly — no database or tools needed.",
            })

        # We intentionally emit NO events for plan / retrieval / inputs / tools
        # so those boxes stay idle in the UI.

        yield ("machine", {
            "node": "compose", "status": "active",
            "title": "Composing Answer", "detail": "Generating response...",
        })

        answer = self._generate_direct_answer(query, attachments, intent)

        yield ("machine", {
            "node": "compose", "status": "done",
            "title": "Composing Answer", "detail": "Response ready.",
        })
        yield ("machine", {
            "node": "output", "status": "active",
            "title": "Streaming", "detail": "Sending response...",
        })

        response = ChatResponse(
            answer=answer,
            supported=True,
            user_inputs={},
            assumed_inputs={},
            assumptions=[],
            sources=[],
            tool_trace=[],
            retrieval_trace=[],
            what_i_used=["Direct response (no pipeline)"],
        )
        yield ("machine", {
            "node": "output", "status": "done",
            "title": "Streaming", "detail": "Complete.",
        })
        yield ("response", response)

    def _generate_direct_answer(
        self,
        query: str,
        attachments: list[Attachment],
        intent: str,
    ) -> str:
        """Produce the text answer for a direct-response intent."""

        # -- Decline: instant, no LLM ----
        if intent == "decline":
            return self._DECLINE_ANSWER

        # -- Greeting ----
        _GREETING_FALLBACK = (
            "Hello! I'm the EC3 Assistant — here to help with structural "
            "engineering and Eurocodes. What can I help you with?"
        )
        if intent == "greeting":
            try:
                raw = self.orchestrator_llm.generate(
                    system_prompt=(
                        "You are the EC3 Assistant, a structural engineering chatbot "
                        "specialising in Eurocodes. The user sent a greeting. "
                        "Reply with a brief, warm greeting (1-2 complete sentences) and "
                        "offer to help with structural engineering questions. "
                        "End with a full stop or question mark. Never end with a comma."
                    ),
                    user_prompt=query,
                    temperature=0.5,
                    max_tokens=500,
                    reasoning_effort="low",
                )
                # Strip trailing whitespace / commas
                cleaned = raw.rstrip().rstrip(",").rstrip()
                if not cleaned or len(cleaned) < 10:
                    return _GREETING_FALLBACK
                # Detect obviously truncated responses (ends with a
                # conjunction, preposition, article, etc.)
                _BROKEN_TAILS = {
                    "and", "or", "but", "the", "a", "an", "to", "for",
                    "with", "in", "on", "at", "of", "is", "am", "are",
                    "i", "we", "you", "that", "this", "my", "your",
                }
                last_word = cleaned.rstrip(".!?,;:").rsplit(None, 1)[-1].lower()
                if last_word in _BROKEN_TAILS:
                    logger.warning("greeting_truncated", extra={"raw": raw[:120]})
                    return _GREETING_FALLBACK
                # Ensure proper ending punctuation
                if cleaned[-1] not in ".!?":
                    cleaned += "."
                return cleaned
            except Exception:
                return _GREETING_FALLBACK

        # -- Answer: conversational engineering response (with vision) ----
        cleaned = self._ATTACHMENT_MARKER_RE.sub("", query).strip()
        has_images = any(a.is_image and a.data_url for a in attachments)
        system_prompt = (
            "You are the EC3 Assistant, a structural engineering chatbot specialising "
            "in Eurocodes. The user asked a question related to structural engineering "
            "that can be answered conversationally — no database lookup or calculator "
            "is needed. If images are provided, describe and interpret them from a "
            "structural engineering perspective (identify elements like beams, columns, "
            "loads, connections, section types, etc.). Be professional, concise, and "
            "helpful. If the user would benefit from a detailed calculation or code "
            "check, suggest they ask a follow-up question so you can run the full "
            "pipeline."
        )
        try:
            if has_images:
                content_parts: list[dict[str, Any]] = []
                for att in attachments:
                    if att.is_image and att.data_url:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": att.data_url},
                        })
                content_parts.append({
                    "type": "text",
                    "text": cleaned or "Describe what you see in this image from a structural engineering perspective.",
                })
                return self.orchestrator_llm.generate_multimodal(
                    system_prompt=system_prompt,
                    content_parts=content_parts,
                    temperature=0.3,
                    max_tokens=4000,
                )
            else:
                return self.orchestrator_llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=cleaned,
                    temperature=0.3,
                    max_tokens=4000,
                )
        except Exception as exc:
            logger.exception("direct_answer_generation_failed")
            return (
                "I understand this is an engineering-related question, but I encountered "
                f"an error generating a response: {exc}\n\nPlease try rephrasing or ask "
                "a more specific question so I can use the full calculation pipeline."
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

        # Use the FIRST user message as anchor — it establishes the base
        # engineering query; subsequent messages modify its parameters.
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
                    max_tokens=1024,
                    reasoning_effort="low",
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
        return f"{query}. {anchor_msg[:200]}"

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

        # Auto-resolve from accumulated session outputs.
        # Computed values OVERRIDE extraction guesses.
        flat = _flatten_tool_outputs(tool_outputs)
        extraction_guesses = set(
            extraction.tool_inputs.get(tool_name, {}).keys()
        ) | set(extraction.assumed_inputs.keys())
        entry = self.tool_registry.get(tool_name)
        if entry:
            props = entry.input_schema.get("properties", {})

            # Phase 1: exact-name match — override with computed values
            phase1_resolved: set[str] = set()
            for param in props:
                if param in flat:
                    base[param] = flat[param]
                    phase1_resolved.add(param)

            # Phase 2: LLM resolution for unresolved + unverified guesses
            needs_llm: list[str] = []
            for p in props:
                if base.get(p) is None:
                    needs_llm.append(p)
                elif p in extraction_guesses and p not in phase1_resolved:
                    needs_llm.append(p)

            if needs_llm and flat and self.orchestrator_llm.available:
                resolved = self._llm_resolve_inputs(
                    tool_name, props, needs_llm, base, tool_outputs,
                )
                base.update(resolved)

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
        for c in retrieved:
            clause_text = c.clause.text.strip()
            clause_evidence.append(f"[{c.clause.clause_id} — {c.clause.clause_title}]: {clause_text}")

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
                    "You are a senior structural engineer answering a colleague's question at a desk review.\n\n"
                    "FORMATTING — CRITICAL:\n"
                    "- All formulas and math expressions MUST be wrapped in dollar signs: $F_{v,Rd}$, $\\frac{a}{b}$, $\\gamma_{M2}$.\n"
                    "- Inline math uses single dollars: $F_{v,Rd} = 94.08$ kN.\n"
                    "- Never write bare LaTeX commands without dollar-sign delimiters.\n"
                    "- Bold **key numerical results**: e.g. **$F_{v,Rd}$ = 94.08 kN**.\n"
                    "- Ensure markdown emphasis is balanced (never leave dangling '**').\n\n"
                    "CONTENT:\n"
                    "1. First sentence MUST directly answer the colleague's question. Start with a capital letter.\n"
                    "   - If they asked for a calculation: state what you calculated, for what parameters, and the result.\n"
                    "     Example: 'For an M20 Grade 8.8 bolt in single shear through the threaded portion, the design shear resistance is **$F_{v,Rd}$ = 94.08 kN** per bolt.'\n"
                    "   - If they asked a conceptual question: lead with a direct answer.\n"
                    "   - If default values were assumed, name them in the first sentence.\n"
                    "2. Explain the method, formula, and key parameters in detail. Show the formula with values substituted in where available.\n"
                    "3. Mention governing EC clauses ONLY when explicit clause evidence is provided below.\n"
                    "4. Use ONLY the provided evidence. Never invent values.\n"
                    "5. If there is no EC clause evidence, do not reference EN 1993 or clause numbers.\n"
                    "6. Write as much detail as the evidence supports. Cover the full engineering context — what the check is, why it matters, what parameters govern the result, and any important caveats or follow-up checks.\n\n"
                    "STRUCTURE:\n"
                    "- Use markdown structure: paragraphs, bullet points, and line breaks to organise the response.\n"
                    "- Break different aspects into separate paragraphs (result, method, parameters, caveats).\n"
                    "- Use bullet points for lists of parameters, checks, or conditions.\n"
                    "- Display key formulas on their own line using $$...$$ display math."
                ),
                user_prompt=(
                    f"Colleague's question: {query}\n\n"
                    f"Retrieved EC3 clauses:\n" + "\n".join(clause_evidence) + "\n\n"
                    f"Tool results:\n{json.dumps(tool_evidence, default=str)}\n\n"
                    "Give a detailed engineering answer. Wrap ALL math in $...$ delimiters."
                ),
                temperature=0.15,
                max_tokens=16384,
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
            text = "Results computed from the available tools and supporting sources."
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
        headline = self._compose_result_headline(
            plan=plan,
            tool_outputs=tool_outputs,
            query=query,
        )
        basis = self._compose_clause_basis_sentence(retrieved=retrieved, tool_outputs=tool_outputs)
        working = narrative
        if not basis:
            working = self._strip_normative_mentions(working)
        return self.response_formatter.polish_narrative(
            working,
            headline=headline,
            basis=basis,
        )

    @staticmethod
    def _strip_normative_mentions(text: str) -> str:
        if not text:
            return text
        cleaned = re.sub(
            r"(?:\(|\b)(?:EC3|EN\s*1993)[^.!?]*?(?:Cl\.\s*[A-Za-z0-9_.()\-]+)?(?:\)|\b)",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

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

        base_key = self.response_formatter.strip_unit_suffix(key)
        symbol = self.response_formatter.pretty_key(base_key)
        value_text = self.response_formatter.format_value(key, value)
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

        sentence = f"{context}the {quantity} is **{symbol} = {value_text}**."
        # Ensure headline starts with a capital letter
        return sentence[0].upper() + sentence[1:] if sentence else ""

    def _compose_clause_basis_sentence(
        self,
        *,
        retrieved: list[RetrievedClause],
        tool_outputs: dict[str, dict[str, Any]],
    ) -> str:
        seen: set[str] = set()
        clause_refs: list[tuple[str, str, str]] = []

        for payload in tool_outputs.values():
            for ref in payload.get("clause_references", []):
                doc_id = str(ref.get("doc_id", "")).strip()
                cid = str(ref.get("clause_id", "")).strip()
                title = str(ref.get("title", "")).strip()
                norm = self._normalize_clause_id(cid)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                clause_refs.append((doc_id, cid, title))
                if len(clause_refs) >= 2:
                    break
            if len(clause_refs) >= 2:
                break

        if len(clause_refs) < 2:
            for item in retrieved:
                doc_id = str(item.clause.doc_id).strip()
                cid = str(item.clause.clause_id).strip()
                title = str(item.clause.clause_title).strip()
                norm = self._normalize_clause_id(cid)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                clause_refs.append((doc_id, cid, title))
                if len(clause_refs) >= 2:
                    break

        normative_refs = [ref for ref in clause_refs if self._is_normative_doc(ref[0])]
        if not normative_refs:
            return ""
        if len(normative_refs) == 1:
            doc_id, cid, title = normative_refs[0]
            standard = self._source_standard_label(doc_id)
            return f"This is checked against {standard}, Cl. {cid} ({title})."

        first_doc, first_cid, first_title = normative_refs[0]
        second_doc, second_cid, second_title = normative_refs[1]
        first_standard = self._source_standard_label(first_doc)
        second_standard = self._source_standard_label(second_doc)
        if first_standard == second_standard:
            return (
                f"This follows {first_standard}, Cl. {first_cid} ({first_title}) "
                f"with supporting classification guidance from Cl. {second_cid} ({second_title})."
            )
        return (
            f"This follows {first_standard}, Cl. {first_cid} ({first_title}) "
            f"with supporting guidance from {second_standard}, Cl. {second_cid} ({second_title})."
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
                    lines.append("<table class=\"tool-io-table\">")
                    lines.append("<thead><tr><th scope=\"col\">Parameter</th><th scope=\"col\">Value</th></tr></thead>")
                    lines.append("<tbody>")
                    for k, v in inputs_used.items():
                        key_cell = self.response_formatter.pretty_key(k)
                        val_cell = self.response_formatter.format_value(k, v)
                        lines.append(f"<tr><td>{key_cell}</td><td>{val_cell}</td></tr>")
                    lines.append("</tbody></table>")
                    lines.append("")

                lines.append("<table class=\"tool-io-table tool-output\">")
                lines.append("<thead><tr><th scope=\"col\">Output</th><th scope=\"col\">Value</th></tr></thead>")
                lines.append("<tbody>")
                for k, v in outputs.items():
                    if isinstance(v, (int, float, str, bool)):
                        key_cell = self.response_formatter.pretty_key(k)
                        val_cell = self.response_formatter.format_value(k, v)
                        lines.append(f"<tr><td>{key_cell}</td><td><strong>{val_cell}</strong></td></tr>")
                lines.append("</tbody></table>")
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
                lines.append("")
                seen_refs: set[str] = set()
                ref_idx = 0
                for s in relevant:
                    ref_key = self._normalize_clause_id(s.clause_id)
                    if ref_key in seen_refs:
                        continue
                    seen_refs.add(ref_key)

                    clause_record = self._lookup_clause(s.doc_id, s.clause_id)
                    ref_idx += 1
                    standard = clause_record.standard if clause_record else self._source_standard_label(s.doc_id)
                    locator = self._format_reference_locator(s.doc_id, s.clause_id)
                    label = f"{standard}, {locator} — {s.clause_title}" if locator else f"{standard} — {s.clause_title}"

                    if clause_record and clause_record.text.strip():
                        text_preview = self._format_clause_text_for_display(clause_record.text)
                        lines.append(f"<details class=\"ref-clause\">")
                        lines.append(f"<summary><strong>{ref_idx}.</strong> {label}</summary>")
                        lines.append("")
                        lines.append(f"<div class=\"clause-text\">")
                        lines.append(text_preview)
                        lines.append("</div>")
                        lines.append("</details>")
                        lines.append("")
                    else:
                        lines.append(f"{ref_idx}. {label}")
                        lines.append("")

        return self.response_formatter.format_markdown("\n".join(lines).strip())

    @staticmethod
    def _normalize_clause_id(clause_id: str) -> str:
        idx = clause_id.find("(")
        return clause_id[:idx].strip() if idx > 0 else clause_id.strip()

    def _lookup_clause(self, doc_id: str, clause_id: str) -> ClauseRecord | None:
        key = (doc_id, clause_id)
        if key in self._clause_lookup:
            return self._clause_lookup[key]
        norm = self._normalize_clause_id(clause_id)
        return self._clause_lookup.get((doc_id, norm))

    def _is_normative_doc(self, doc_id: str) -> bool:
        if not doc_id:
            return False
        if doc_id in self._document_lookup:
            return True
        return doc_id.lower().startswith("ec3.")

    def _source_standard_label(self, doc_id: str) -> str:
        entry = self._document_lookup.get(doc_id)
        if entry:
            return entry.standard
        if doc_id.lower().startswith("ec3."):
            lower = doc_id.lower()
            match = re.search(r"(?:en)?(1993-\d-\d)", lower)
            if match:
                return f"EN {match.group(1)}"
            return "EN 1993"
        cleaned = doc_id.replace("_", " ").strip()
        return cleaned if cleaned else "Source"

    def _format_reference_locator(self, doc_id: str, clause_id: str) -> str:
        cid = clause_id.strip()
        if not cid:
            return ""
        if self._is_normative_doc(doc_id) and re.match(r"^\d", cid):
            return f"Cl. {cid}"
        return cid

    def _format_clause_text_for_display(self, text: str) -> str:
        """Format clause text for clean display: escape HTML, preserve paragraphs."""
        t = (text or "").strip()
        if not t:
            return ""
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        paragraphs = re.split(r"\n\s*\n", t)
        blocks = [f"<p class=\"clause-p\">{p.strip().replace(chr(10), '<br>')}</p>" for p in paragraphs if p.strip()]
        return "\n".join(blocks)

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
        retrieval_summary = (
            f"Retrieval: {len(retrieval_trace)} search pass(es)"
            if retrieval_trace
            else "Retrieval: skipped"
        )
        summaries = [f"Plan: {plan.mode} — {plan.rationale}", retrieval_summary]
        if tool_trace:
            chain = " → ".join(s.tool_name for s in tool_trace)
            summaries.append(f"Tool chain: {chain}")
        return summaries
