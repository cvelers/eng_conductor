"""Agentic task-decomposition loop.

Replaces the rigid pipeline in ``engine.py`` with a decompose-then-execute
architecture.  Complex queries are broken into individual tasks, each
executed with its own focused search → calculate → compose cycle.

The key insight: each compose call gets a **clean, isolated context** with
only the relevant clauses and tool outputs — maximising LLM intelligence
per sub-task instead of diluting it across everything at once.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from backend.config import Settings
from backend.orchestrator.core import (
    CentralIntelligenceOrchestrator,
    PlanResult,
    _flatten_tool_outputs,
)
from backend.schemas import (
    Attachment,
    ChatResponse,
    Citation,
    RetrievalTraceStep,
    ToolTraceStep,
)
from backend.utils.citations import build_citation_address
from backend.utils.json_utils import parse_json_loose

logger = logging.getLogger(__name__)


MAX_TOOL_RETRIES = 2


@dataclass
class _TaskSpec:
    """One decomposed sub-task."""

    summary: str
    query: str
    search_query: str
    needs_search: bool = True
    tools: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


# ── Compose system prompt — same engineering formatting as engine.py ──

_COMPOSE_SYSTEM = (
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
    "   - If they asked a conceptual question: lead with a direct answer.\n"
    "   - If default values were assumed, name them in the first sentence.\n"
    "2. Explain the method, formula, and key parameters in detail. Show the formula with values substituted in where available.\n"
    "3. Mention governing EC clauses ONLY when explicit clause evidence is provided below.\n"
    "4. Use ONLY the provided evidence. Never invent values.\n"
    "5. If there is no EC clause evidence, do not reference EN 1993 or clause numbers.\n"
    "6. Write as much detail as the evidence supports.\n\n"
    "STRUCTURE:\n"
    "- Use markdown structure: paragraphs, bullet points, and line breaks.\n"
    "- Break different aspects into separate paragraphs.\n"
    "- Display key formulas on their own line using $$...$$ display math."
)

_DECOMPOSE_SYSTEM = (
    "You decompose user engineering queries into individual, independent sub-tasks.\n"
    "Each sub-task should be self-contained and addressable with a focused search + calculation.\n\n"
    "Return ONLY a JSON array of task objects. Each task:\n"
    "{\n"
    '  "summary": "Short description (shown as plan step)",\n'
    '  "query": "Focused technical query for this task",\n'
    '  "search_query": "Search terms for finding relevant EC3 clauses",\n'
    '  "needs_search": true/false,\n'
    '  "tools": ["tool_name_1", "tool_name_2"],\n'
    '  "inputs": {"param1": value1, ...}\n'
    "}\n\n"
    "Rules:\n"
    "- If the query asks for only ONE thing, return a single-element array.\n"
    "- Only split when there are genuinely separate engineering questions.\n"
    "- Order tasks logically (dependencies first).\n"
    "- Only include tools that are directly relevant.\n"
    "- In 'inputs', only include values the user explicitly states in their query.\n"
    "- NEVER guess or invent values for parameters that an earlier task's tool\n"
    "  will compute (e.g. section_class from section_classification, fy_mpa from\n"
    "  steel_grade_properties).  Omit those parameters entirely — the orchestrator\n"
    "  will resolve them automatically from the session context.\n"
    "- Return valid JSON only, no markdown fences, no explanation."
)


class AgentLoop:
    """Agentic task-decomposition orchestrator.

    Wraps ``CentralIntelligenceOrchestrator`` via composition to reuse
    intent classification, tool execution, source collection, formatting,
    etc.  Only the top-level ``run_stream`` is replaced.
    """

    def __init__(
        self,
        *,
        orchestrator: CentralIntelligenceOrchestrator,
        settings: Settings,
    ) -> None:
        self.cio = orchestrator
        self.settings = settings
        self._current_tasks: list[_TaskSpec] = []

    # ── Public API ──────────────────────────────────────────────────

    def run(
        self,
        query: str,
        *,
        history: list | None = None,
        thinking_mode: str = "thinking",
        attachments: list[Attachment] | None = None,
        is_edit: bool = False,
    ) -> ChatResponse:
        final: ChatResponse | None = None
        for _etype, payload in self.run_stream(
            query,
            history=history,
            thinking_mode=thinking_mode,
            attachments=attachments,
            is_edit=is_edit,
        ):
            if isinstance(payload, ChatResponse):
                final = payload
        if final is None:
            raise RuntimeError("AgentLoop did not produce a final response.")
        return final

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
        query = (
            raw_query
            if is_edit
            else self.cio._resolve_followup(raw_query, history or [])
        )
        selected_mode = self.cio._normalize_thinking_mode(thinking_mode)

        # ── Phase 0: Intent classification ──────────────────────────
        classification = self.cio._classify_intent(query, attachments)
        intent = classification["intent"]
        logger.info(
            "agent_intent_classified",
            extra={"intent": intent, "query_preview": query[:80]},
        )

        if intent in ("decline", "greeting", "answer"):
            # Direct-response path — forward all events (machine + response)
            yield from self.cio._handle_direct_response(query, attachments, intent)
            return

        # ── Flow graph: intake ────────────────────────────────────────
        yield ("machine", {"node": "intake", "status": "active", "title": "Query Intake", "detail": "Analyzing your question..."})
        yield ("machine", {"node": "intake", "status": "done", "title": "Query Intake", "detail": "Planning tasks..."})

        # ── Phase 1: Task decomposition ─────────────────────────────
        yield ("thinking", {"content": "Analyzing your question..."})

        tasks = self._decompose(query, selected_mode)
        self._current_tasks = tasks  # Store for plan context in LLM resolution
        logger.info(
            "agent_tasks_decomposed",
            extra={"task_count": len(tasks), "summaries": [t.summary for t in tasks]},
        )

        yield (
            "plan",
            {
                "steps": [
                    {"id": i, "text": t.summary, "status": "pending"}
                    for i, t in enumerate(tasks)
                ]
            },
        )

        # ── Phase 2: Execute each task sequentially ─────────────────
        all_sources: list[Citation] = []
        all_tool_trace: list[ToolTraceStep] = []
        all_retrieval_trace: list[RetrievalTraceStep] = []
        all_retrieved: list = []
        all_tool_outputs: dict[str, dict[str, Any]] = {}
        answer_parts: list[str] = []
        all_user_inputs: dict[str, Any] = {}
        all_assumed_inputs: dict[str, Any] = {}
        all_assumptions: list[str] = []

        for i, task in enumerate(tasks):
            yield ("plan_update", {"step_id": i, "status": "in_progress"})

            # 2a. SEARCH — focused retrieval for this task only
            task_clauses = []
            if task.needs_search:
                yield ("machine", {"node": "retrieval", "status": "active", "title": "Retrieving", "detail": f"Searching: {task.search_query[:60]}"})
                yield (
                    "tool_start",
                    {"tool": "search", "args": {"query": task.search_query}},
                )
                try:
                    results, trace = self.cio.retriever.retrieve(
                        task.search_query,
                        top_k=self._top_k_for_mode(selected_mode),
                    )
                    task_clauses = results
                    all_retrieved.extend(results)
                    all_retrieval_trace.extend(
                        [
                            RetrievalTraceStep(
                                iteration=s.get("iteration", 0),
                                query=str(s.get("query", task.search_query)),
                                top_clause_ids=s.get("top_clause_ids", []),
                            )
                            for s in trace
                            if isinstance(s, dict)
                        ]
                    )
                    clause_summaries = [
                        {
                            "clause_id": r.clause.clause_id,
                            "title": r.clause.clause_title,
                        }
                        for r in results[:6]
                    ]
                    top_meta = [{"doc_id": r.clause.doc_id, "clause_id": r.clause.clause_id} for r in results[:6]]
                    yield ("machine", {"node": "retrieval", "status": "done", "title": "Retrieved", "detail": f"Found {len(results)} clauses", "meta": {"top": top_meta}})
                    yield (
                        "tool_result",
                        {
                            "tool": "search",
                            "status": "ok",
                            "summary": f"Found {len(results)} relevant clauses",
                            "result": {"clauses": clause_summaries},
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "agent_search_failed",
                        extra={"error": str(exc), "query": task.search_query},
                    )
                    yield ("machine", {"node": "retrieval", "status": "error", "title": "Retrieval", "detail": f"Search failed: {exc}"})
                    yield (
                        "tool_result",
                        {
                            "tool": "search",
                            "status": "error",
                            "summary": f"Search failed: {exc}",
                            "result": {},
                        },
                    )

            # 2b. CALCULATE — tools needed for this task only
            task_tool_outputs: dict[str, dict[str, Any]] = {}
            tool_failures: list[str] = []
            if task.tools:
                ordered = self.cio._normalize_tool_chain(
                    task.tools,
                    already_run=set(all_tool_outputs.keys()),
                )
                yield ("machine", {"node": "tools", "status": "active", "title": "Tools", "detail": "Executing tool chain."})
                for tool_name in ordered:
                    inputs = self._build_task_tool_inputs(
                        tool_name, task, all_tool_outputs
                    )
                    success = False
                    last_error = ""
                    for attempt in range(1 + MAX_TOOL_RETRIES):
                        yield (
                            "tool_start",
                            {
                                "tool": tool_name,
                                "args": inputs,
                            },
                        )
                        try:
                            payload = self.cio.tool_runner.run(tool_name, inputs)
                            result = payload.get("result", {})
                            task_tool_outputs[tool_name] = result
                            all_tool_outputs[tool_name] = result
                            step = ToolTraceStep(
                                tool_name=tool_name,
                                status="ok",
                                inputs=inputs,
                                outputs=result.get("outputs", result),
                            )
                            all_tool_trace.append(step)

                            # Compact summary for the UI
                            outputs = result.get("outputs", {})
                            summary_parts = []
                            for k, v in outputs.items():
                                if isinstance(v, (int, float)):
                                    summary_parts.append(
                                        f"{k.replace('_', ' ')} = {v}"
                                    )
                            summary = "; ".join(summary_parts[:3]) or "Complete"

                            yield ("machine", {"node": "tools", "status": "active", "title": "Tools", "detail": f"Tool {tool_name} completed.", "meta": {"tool": tool_name}})
                            yield (
                                "tool_result",
                                {
                                    "tool": tool_name,
                                    "status": "ok",
                                    "summary": summary,
                                    "result": result,
                                },
                            )
                            success = True
                            break
                        except Exception as exc:
                            last_error = str(exc)
                            yield (
                                "tool_result",
                                {
                                    "tool": tool_name,
                                    "status": "error",
                                    "summary": f"Failed (attempt {attempt + 1}): {exc}",
                                    "result": {},
                                },
                            )
                            if attempt < MAX_TOOL_RETRIES:
                                # Ask LLM to fix inputs
                                fixed = self._fix_tool_inputs(
                                    tool_name, inputs, last_error, task
                                )
                                if fixed and fixed != inputs:
                                    inputs = fixed
                                    yield ("machine", {"node": "tools", "status": "active", "title": "Tools", "detail": f"Retrying {tool_name} with corrected inputs..."})
                                else:
                                    break  # LLM couldn't fix — stop retrying

                    if not success:
                        step = ToolTraceStep(
                            tool_name=tool_name,
                            status="error",
                            inputs=inputs,
                            error=last_error,
                        )
                        all_tool_trace.append(step)
                        tool_failures.append(f"{tool_name}: {last_error}")
                        yield ("machine", {"node": "tools", "status": "error", "title": "Tools", "detail": f"Tool {tool_name} failed after retries.", "meta": {"tool": tool_name}})

                yield ("machine", {"node": "tools", "status": "done", "title": "Tools", "detail": "Tool execution finished."})

            # Track inputs/assumptions from task
            if task.inputs:
                all_user_inputs.update(task.inputs)

            # 2c. COMPOSE — fresh LLM call with ONLY this task's context
            yield ("machine", {"node": "compose", "status": "active", "title": "Composing Answer", "detail": f"Composing: {task.summary[:60]}"})
            yield (
                "thinking",
                {"content": f"Composing answer for: {task.summary}"},
            )

            partial = self._compose_task_answer(
                task=task,
                original_query=query,
                retrieved=task_clauses,
                tool_outputs=task_tool_outputs,
                tool_failures=tool_failures,
                thinking_mode=selected_mode,
            )

            # Stream the partial answer in natural chunks
            for chunk in _chunk_naturally(partial):
                yield ("delta", {"content": chunk})

            answer_parts.append(partial)

            # Collect sources for this task
            task_sources = self.cio._collect_sources(
                task_clauses, task_tool_outputs
            )
            all_sources.extend(task_sources)

            yield ("machine", {"node": "compose", "status": "done", "title": "Composing Answer", "detail": "Composition complete."})
            yield ("plan_update", {"step_id": i, "status": "done"})

        # ── Phase 3: Finalize ───────────────────────────────────────
        yield ("machine", {"node": "output", "status": "active", "title": "Streaming", "detail": "Building final response..."})

        full_answer = "\n\n---\n\n".join(answer_parts) if len(answer_parts) > 1 else (answer_parts[0] if answer_parts else "")

        # Format answer through the response formatter
        full_answer = self.cio.response_formatter.format_markdown(full_answer)

        # Append tool detail tables, assumptions, and references
        appendix = self._build_answer_appendix(
            tool_outputs=all_tool_outputs,
            tool_trace=all_tool_trace,
            assumptions=all_assumptions,
            sources=all_sources,
            retrieved=all_retrieved,
            narrative=full_answer,
        )
        if appendix:
            full_answer = full_answer + "\n\n" + appendix

        # Deduplicate sources
        seen_addresses: set[str] = set()
        unique_sources: list[Citation] = []
        for s in all_sources:
            if s.citation_address not in seen_addresses:
                seen_addresses.add(s.citation_address)
                unique_sources.append(s)

        what_i_used = self._build_what_i_used(tasks, all_retrieval_trace, all_tool_trace)

        response = ChatResponse(
            answer=full_answer,
            supported=bool(unique_sources or all_tool_trace),
            user_inputs=all_user_inputs,
            assumed_inputs=all_assumed_inputs,
            assumptions=all_assumptions,
            sources=unique_sources,
            tool_trace=all_tool_trace,
            retrieval_trace=all_retrieval_trace,
            what_i_used=what_i_used,
        )
        yield ("machine", {"node": "output", "status": "done", "title": "Streaming", "detail": "Complete."})
        yield ("response", response)

    # ── Answer appendix ─────────────────────────────────────────────

    def _build_answer_appendix(
        self,
        *,
        tool_outputs: dict[str, dict[str, Any]],
        tool_trace: list[ToolTraceStep],
        assumptions: list[str],
        sources: list[Citation],
        retrieved: list,
        narrative: str,
    ) -> str:
        """Build the structured appendix (tool tables, assumptions, references).

        Mirrors the appendix sections from
        ``CentralIntelligenceOrchestrator._build_markdown_answer()``.
        """
        lines: list[str] = []

        # ── Tool detail tables ──
        if tool_outputs:
            for step in tool_trace:
                if step.status != "ok":
                    continue
                payload = tool_outputs.get(step.tool_name)
                if not payload:
                    continue
                outputs = payload.get("outputs", {})
                inputs_used = payload.get("inputs_used", {})
                if not outputs:
                    continue

                pretty = self.cio._pretty_tool_name(step.tool_name)
                lines.append(f'<details class="tool-result">')
                lines.append(f"<summary><strong>{pretty}</strong> detailed results</summary>")
                lines.append("")

                if inputs_used:
                    lines.append('<table class="tool-io-table">')
                    lines.append('<thead><tr><th scope="col">Parameter</th><th scope="col">Value</th></tr></thead>')
                    lines.append("<tbody>")
                    for k, v in inputs_used.items():
                        key_cell = self.cio.response_formatter.pretty_key(k)
                        val_cell = self.cio.response_formatter.format_value(k, v)
                        lines.append(f"<tr><td>{key_cell}</td><td>{val_cell}</td></tr>")
                    lines.append("</tbody></table>")
                    lines.append("")

                lines.append('<table class="tool-io-table tool-output">')
                lines.append('<thead><tr><th scope="col">Output</th><th scope="col">Value</th></tr></thead>')
                lines.append("<tbody>")
                for k, v in outputs.items():
                    if isinstance(v, (int, float, str, bool)):
                        key_cell = self.cio.response_formatter.pretty_key(k)
                        val_cell = self.cio.response_formatter.format_value(k, v)
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

        # ── Assumptions ──
        if assumptions:
            lines.append('<details class="assumptions">')
            lines.append("<summary>Assumptions used</summary>")
            lines.append("")
            for a in assumptions:
                lines.append(f"- {a}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # ── Tool errors ──
        if any(s.status == "error" for s in tool_trace):
            lines.append("\n**Tool Errors:**\n")
            for s in tool_trace:
                if s.status == "error":
                    lines.append(f"- {s.tool_name}: {s.error}")

        # ── References ──
        if sources:
            try:
                filtered = self.cio._select_relevant_sources(
                    narrative=narrative, sources=sources,
                    retrieved=retrieved, tool_outputs=tool_outputs,
                )
            except Exception:
                filtered = sources
            if not filtered:
                filtered = sources
            relevant = [
                s for s in filtered
                if s.clause_id and s.clause_id != "0"
                and s.clause_title and s.clause_title != "text"
            ]
            if relevant:
                lines.append("\n---\n")
                lines.append("**References:**")
                lines.append("")
                seen_refs: set[str] = set()
                ref_idx = 0
                for s in relevant:
                    ref_key = self.cio._normalize_clause_id(s.clause_id)
                    if ref_key in seen_refs:
                        continue
                    seen_refs.add(ref_key)

                    clause_record = self.cio._lookup_clause(s.doc_id, s.clause_id)
                    ref_idx += 1
                    standard = clause_record.standard if clause_record else self.cio._source_standard_label(s.doc_id)
                    locator = self.cio._format_reference_locator(s.doc_id, s.clause_id)
                    label = f"{standard}, {locator} — {s.clause_title}" if locator else f"{standard} — {s.clause_title}"

                    if clause_record and clause_record.text.strip():
                        text_preview = self.cio._format_clause_text_for_display(clause_record.text)
                        lines.append(f'<details class="ref-clause">')
                        lines.append(f"<summary><strong>{ref_idx}.</strong> {label}</summary>")
                        lines.append("")
                        lines.append(f'<div class="clause-text">')
                        lines.append(text_preview)
                        lines.append("</div>")
                        lines.append("</details>")
                        lines.append("")
                    else:
                        lines.append(f"{ref_idx}. {label}")
                        lines.append("")

        result = "\n".join(lines).strip()
        return self.cio.response_formatter.format_markdown(result) if result else ""

    # ── Task decomposition ──────────────────────────────────────────

    def _decompose(self, query: str, thinking_mode: str) -> list[_TaskSpec]:
        """Break query into focused sub-tasks."""

        # Standard mode: never decompose
        if thinking_mode == "standard":
            return [self._single_task(query)]

        # Try LLM decomposition
        if self.cio.orchestrator_llm.available:
            try:
                return self._llm_decompose(query, thinking_mode)
            except Exception as exc:
                logger.warning(
                    "agent_decompose_failed",
                    extra={"error": str(exc)},
                )

        # Heuristic fallback
        return [self._single_task(query)]

    def _llm_decompose(self, query: str, thinking_mode: str) -> list[_TaskSpec]:
        valid_tools = list(self.cio.tool_registry.keys())
        tool_list = "\n".join(
            f"- {name}: {self.cio.tool_registry[name].description} "
            f"| inputs: {list(self.cio.tool_registry[name].input_schema.get('properties', {}).keys())}"
            for name in valid_tools
        )

        doc_list = "\n".join(
            f"- {e.standard} ({e.year_version}): {e.title}"
            for e in self.cio.document_registry
        )
        if not doc_list:
            doc_list = "(no documents loaded)"

        mode_hint = ""
        if thinking_mode == "extended":
            mode_hint = "\nExtended mode: be thorough. Include separate search and calculation tasks where appropriate."

        raw = self.cio.orchestrator_llm.generate(
            system_prompt=_DECOMPOSE_SYSTEM,
            user_prompt=(
                f"User query: {query}\n\n"
                f"Available calculator tools:\n{tool_list}\n\n"
                f"Available Eurocode documents:\n{doc_list}\n"
                f"{mode_hint}\n\n"
                "Return JSON array of tasks."
            ),
            temperature=0,
            max_tokens=2048,
            reasoning_effort="low" if thinking_mode != "extended" else None,
        )

        parsed = parse_json_loose(raw)
        if not isinstance(parsed, list):
            # Maybe it's a single object
            if isinstance(parsed, dict):
                parsed = [parsed]
            else:
                raise ValueError(f"Expected JSON array, got: {type(parsed)}")

        tasks: list[_TaskSpec] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            tools = [t for t in item.get("tools", []) if t in self.cio.tool_registry]
            tasks.append(
                _TaskSpec(
                    summary=str(item.get("summary", "Task")),
                    query=str(item.get("query", query)),
                    search_query=str(item.get("search_query", item.get("query", query))),
                    needs_search=bool(item.get("needs_search", True)),
                    tools=tools,
                    inputs=item.get("inputs", {}),
                )
            )

        if not tasks:
            return [self._single_task(query)]

        return tasks

    def _single_task(self, query: str) -> _TaskSpec:
        """Build a single task from heuristic analysis (no decomposition)."""
        valid_tools = list(self.cio.tool_registry.keys())
        matched_tools = self.cio._match_tools_for_query(
            query=query, valid_tools=valid_tools
        )
        intent = self.cio._query_intent(query)

        needs_search = not intent.get("pure_calculation", False)
        tools = matched_tools if intent.get("has_calc_intent") or intent.get("has_specific_values") else []

        # Extract concrete values from query for tool inputs
        inputs = self._extract_inputs_from_query(query)

        return _TaskSpec(
            summary=_summarize_query(query),
            query=query,
            search_query=query,
            needs_search=needs_search,
            tools=tools,
            inputs=inputs,
        )

    def _extract_inputs_from_query(self, query: str) -> dict[str, Any]:
        """Pull concrete engineering values from the query text."""
        inputs: dict[str, Any] = {}
        lowered = query.lower()

        # Section name
        m = re.search(r"\b((?:ipe|hea|heb|hem)\s*\d+)\b", lowered, re.IGNORECASE)
        if m:
            inputs["section_name"] = m.group(1).upper().replace(" ", "")

        # Steel grade
        m = re.search(r"\b(s(?:235|275|355|420|460))\b", lowered, re.IGNORECASE)
        if m:
            inputs["steel_grade"] = m.group(1).upper()

        # Bolt diameter
        m = re.search(r"\bm(\d+)\b", lowered)
        if m:
            inputs["bolt_diameter_mm"] = int(m.group(1))

        # Bolt grade
        m = re.search(r"\b(4\.6|4\.8|5\.6|5\.8|6\.8|8\.8|10\.9|12\.9)\b", lowered)
        if m:
            inputs["bolt_grade"] = m.group(1)

        # Span
        m = re.search(r"(\d+(?:\.\d+)?)\s*m(?:\s+span|\b)", lowered)
        if m:
            inputs["span_m"] = float(m.group(1))

        # UDL
        m = re.search(r"(\d+(?:\.\d+)?)\s*kn/m\b", lowered)
        if m:
            inputs["udl_kn_m"] = float(m.group(1))

        # Point load
        m = re.search(r"(\d+(?:\.\d+)?)\s*kn\b(?!\s*/)", lowered)
        if m and "point" in lowered:
            inputs["point_load_kn"] = float(m.group(1))

        return inputs

    # ── Task-level tool input building ──────────────────────────────

    def _build_task_tool_inputs(
        self,
        tool_name: str,
        task: _TaskSpec,
        all_tool_outputs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Build inputs for a tool using the full session context.

        Resolution:
        1. Values extracted from the user query (task.inputs) + settings defaults
        2. LLM resolves ALL expected params when prior tool outputs exist
        """
        # Start with task-level extracted inputs
        base = dict(task.inputs)

        # Add defaults from settings
        if "section_name" not in base:
            base["section_name"] = self.settings.default_section_name
        if "steel_grade" not in base:
            base["steel_grade"] = self.settings.default_steel_grade
        if "gamma_m0" not in base:
            base["gamma_m0"] = self.settings.default_gamma_m0

        entry = self.cio.tool_registry.get(tool_name)
        if entry:
            props = entry.input_schema.get("properties", {})
            expected = set(props.keys())

            # LLM resolves ALL expected params when prior outputs exist
            flat = _flatten_tool_outputs(all_tool_outputs)
            if flat and self.cio.orchestrator_llm.available:
                # Build plan context for LLM
                plan_context = None
                if self._current_tasks:
                    plan_context = [
                        {
                            "step": idx + 1,
                            "summary": t.summary,
                            "tools": t.tools,
                        }
                        for idx, t in enumerate(self._current_tasks)
                    ]
                resolved = self.cio._llm_resolve_inputs(
                    tool_name, props, list(expected), base,
                    all_tool_outputs, plan_context=plan_context,
                )
                base.update(resolved)

            # Filter to expected params
            if expected:
                base = {k: v for k, v in base.items() if k in expected and v is not None}
            else:
                base = {k: v for k, v in base.items() if v is not None}

            # Normalize enum values (e.g. "UDL" → "udl")
            for k, v in list(base.items()):
                if isinstance(v, str) and k in props:
                    allowed = props[k].get("enum")
                    if allowed and v not in allowed:
                        lower = v.lower()
                        if lower in allowed:
                            base[k] = lower
        else:
            base = {k: v for k, v in base.items() if v is not None}

        return base

    def _fix_tool_inputs(
        self,
        tool_name: str,
        failed_inputs: dict[str, Any],
        error_msg: str,
        task: _TaskSpec,
    ) -> dict[str, Any] | None:
        """Ask the LLM to correct tool inputs after a failure."""
        if not self.cio.orchestrator_llm.available:
            return None

        entry = self.cio.tool_registry.get(tool_name)
        schema_desc = ""
        if entry:
            props = entry.input_schema.get("properties", {})
            required = entry.input_schema.get("required", [])
            schema_desc = json.dumps(
                {"properties": props, "required": required}, indent=2
            )

        try:
            raw = self.cio.orchestrator_llm.generate(
                system_prompt=(
                    "You are a tool-input repair agent. A calculator tool failed due "
                    "to invalid inputs. Return ONLY a JSON object containing the "
                    "parameters that need to CHANGE to fix the error. Do NOT include "
                    "parameters that are already correct — only the ones that caused "
                    "the failure or need updating. No explanation."
                ),
                user_prompt=(
                    f"Tool: {tool_name}\n"
                    f"User query: {task.query}\n"
                    f"Input schema:\n{schema_desc}\n\n"
                    f"Failed inputs: {json.dumps(failed_inputs, default=str)}\n"
                    f"Error: {error_msg}\n\n"
                    "Return ONLY the parameters that need to change as JSON."
                ),
                temperature=0,
                max_tokens=1024,
                reasoning_effort="low",
            )
            fixed = parse_json_loose(raw)
            if isinstance(fixed, dict):
                # Merge corrections into original inputs so that
                # values resolved by the LLM (e.g. section_class)
                # are preserved when the fix only changes a subset
                # of parameters (e.g. section_name).
                merged = {**failed_inputs, **fixed}
                # Re-apply enum normalization
                if entry:
                    props = entry.input_schema.get("properties", {})
                    for k, v in list(merged.items()):
                        if isinstance(v, str) and k in props:
                            allowed = props[k].get("enum")
                            if allowed and v not in allowed:
                                lower = v.lower()
                                if lower in allowed:
                                    merged[k] = lower
                return merged
        except Exception as exc:
            logger.warning("fix_tool_inputs_failed", extra={"error": str(exc)})

        return None

    # ── Task-level composition ──────────────────────────────────────

    def _compose_task_answer(
        self,
        *,
        task: _TaskSpec,
        original_query: str,
        retrieved: list,
        tool_outputs: dict[str, dict[str, Any]],
        tool_failures: list[str] | None = None,
        thinking_mode: str,
    ) -> str:
        """Compose a focused answer for one task with isolated context."""

        if not self.cio.orchestrator_llm.available:
            return self._fallback_compose(task, retrieved, tool_outputs)

        # Build clause evidence — only this task's clauses
        clause_evidence = []
        for r in retrieved:
            text = r.clause.text.strip()
            clause_evidence.append(
                f"[{r.clause.clause_id} — {r.clause.clause_title}]: {text}"
            )

        # Build tool evidence — only this task's outputs
        tool_evidence: dict[str, Any] = {}
        for tname, tout in tool_outputs.items():
            tool_evidence[tname] = {
                "inputs_used": tout.get("inputs_used", {}),
                "outputs": tout.get("outputs", {}),
                "notes": tout.get("notes", []),
            }

        clause_text = "\n".join(clause_evidence) if clause_evidence else "(no clauses retrieved)"
        tool_text = json.dumps(tool_evidence, default=str) if tool_evidence else "(no tool results)"

        is_multi_task = task.query != original_query
        context_note = (
            f"This is one part of a multi-part question. The full question was: {original_query}\n"
            f"Focus your answer on this specific sub-task: {task.query}\n"
            if is_multi_task
            else ""
        )

        # If tools failed and we have no tool outputs, tell the LLM honestly
        failure_note = ""
        if tool_failures and not tool_outputs:
            failure_note = (
                "\n\nIMPORTANT: The required calculator tools failed after multiple "
                "retries. You MUST NOT answer the calculation from general knowledge. "
                "Instead, explain that the calculation could not be completed right now "
                "due to a tool error, apologise, and suggest the user try again or "
                "rephrase their query. Be specific about what went wrong.\n"
                f"Tool errors: {'; '.join(tool_failures)}\n"
            )

        reasoning_effort = None
        if thinking_mode == "standard":
            reasoning_effort = "low"
        elif thinking_mode == "extended":
            reasoning_effort = "high"

        try:
            raw = self.cio.orchestrator_llm.generate(
                system_prompt=_COMPOSE_SYSTEM,
                user_prompt=(
                    f"{context_note}"
                    f"Colleague's question: {task.query}\n\n"
                    f"Retrieved EC3 clauses:\n{clause_text}\n\n"
                    f"Tool results:\n{tool_text}\n\n"
                    f"{failure_note}"
                    "Give a detailed engineering answer. Wrap ALL math in $...$ delimiters."
                ),
                temperature=0.15,
                max_tokens=8000,
                reasoning_effort=reasoning_effort,
            )
            return raw.strip()
        except Exception as exc:
            logger.warning(
                "agent_compose_failed",
                extra={"error": str(exc), "task": task.summary},
            )
            return self._fallback_compose(task, retrieved, tool_outputs)

    def _fallback_compose(
        self,
        task: _TaskSpec,
        retrieved: list,
        tool_outputs: dict[str, dict[str, Any]],
    ) -> str:
        """Build a basic answer without LLM."""
        parts: list[str] = []

        if tool_outputs:
            for tname, payload in tool_outputs.items():
                outputs = payload.get("outputs", {})
                if outputs:
                    pretty = tname.replace("_ec3", "").replace("_", " ").title()
                    parts.append(f"**{pretty}** results:")
                    for k, v in outputs.items():
                        if isinstance(v, (int, float, str, bool)):
                            parts.append(f"- {k}: **{v}**")

        if retrieved:
            parts.append("\n**Relevant clauses:**")
            for r in retrieved[:4]:
                parts.append(
                    f"- Cl. {r.clause.clause_id}: {r.clause.clause_title}"
                )

        if not parts:
            parts.append(
                "I processed your request but could not generate a detailed answer. "
                "Please check the tool results above for the raw calculation output."
            )

        return "\n".join(parts)

    # ── Utilities ───────────────────────────────────────────────────

    def _top_k_for_mode(self, mode: str) -> int:
        if mode == "standard":
            return 4
        if mode == "extended":
            return 10
        return self.settings.top_k_clauses

    def _build_what_i_used(
        self,
        tasks: list[_TaskSpec],
        retrieval_trace: list[RetrievalTraceStep],
        tool_trace: list[ToolTraceStep],
    ) -> list[str]:
        summaries = [f"Agent mode: {len(tasks)} task(s)"]
        for i, t in enumerate(tasks):
            summaries.append(f"Task {i + 1}: {t.summary}")
        if retrieval_trace:
            summaries.append(f"Retrieval: {len(retrieval_trace)} search pass(es)")
        if tool_trace:
            chain = " → ".join(s.tool_name for s in tool_trace)
            summaries.append(f"Tool chain: {chain}")
        return summaries


# ── Module-level helpers ────────────────────────────────────────────


def _chunk_naturally(text: str, target: int = 80) -> list[str]:
    """Split text at paragraph/sentence boundaries for natural streaming."""
    if not text:
        return []

    chunks: list[str] = []

    # First split by paragraphs
    paragraphs = re.split(r"\n\n+", text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= target * 2:
            # Small paragraph — yield as-is (with trailing newlines)
            chunks.append(para + "\n\n")
        else:
            # Large paragraph — split by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                if current and len(current) + len(sent) > target:
                    chunks.append(current)
                    current = sent + " "
                else:
                    current += sent + " "
            if current.strip():
                chunks.append(current.strip() + "\n\n")

    return chunks


def _summarize_query(query: str) -> str:
    """Create a short summary of a query for display as a plan step."""
    cleaned = query.strip()
    if len(cleaned) <= 60:
        return cleaned
    # Truncate at a word boundary
    truncated = cleaned[:57]
    last_space = truncated.rfind(" ")
    if last_space > 30:
        truncated = truncated[:last_space]
    return truncated + "..."
