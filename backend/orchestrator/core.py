"""Orchestrator core — shared services and utilities.

Provides intent classification, follow-up resolution, tool-chain management,
source collection, and formatting utilities used by the agentic
task-decomposition loop in ``agent_loop.py``.

The linear pipeline that previously lived in ``engine.py`` has been removed;
only the agent-loop execution path is supported.
"""

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
from backend.schemas import Attachment, ChatResponse, Citation
from backend.tools.response_formatter import ResponseFormatterTool
from backend.tools.runner import MCPToolRunner
from backend.utils.citations import build_citation_address
from backend.utils.json_utils import parse_json_loose

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
    """Shared orchestrator providing services to the agent loop.

    Holds the LLM provider, retriever, tool runner, registries, and
    exposes intent classification, tool-chain resolution, source
    collection, and formatting helpers.
    """

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

    # ── Planning helpers ───────────────────────────────────────────

    def _normalize_thinking_mode(self, mode: str | None) -> str:
        normalized = str(mode or "thinking").strip().lower().replace("-", "_")
        if normalized in _THINKING_MODES:
            return normalized
        return "thinking"

    # ── Query analysis ─────────────────────────────────────────────

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

        has_specific_values = bool(re.search(
            r"\b(?:"
            r"(?:ipe|heb|hea|hem|ub|uc|chs|rhs|shs)\s*\d+"
            r"|s(?:235|275|355|420|460)\b"
            r"|m(?:12|14|16|20|22|24|27|30|36)\b"
            r"|(?:4\.6|4\.8|5\.6|5\.8|6\.8|8\.8|10\.9|12\.9)"
            r"|\d+(?:\.\d+)?\s*(?:mm|cm|kn|mpa|n/mm|knm)"
            r")",
            lowered,
        ))

        lookup_only = has_lookup_intent and not has_calc_intent and not has_numeric
        pure_calculation = (
            has_calc_intent
            and has_specific_values
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

    # ── Tool chain resolution ──────────────────────────────────────

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

    # ── LLM-based session resolution ───────────────────────────────

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
        plan_context: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Use the LLM to resolve ALL tool inputs from session context.

        The LLM is the primary resolver — it sees the full execution plan,
        all prior tool outputs, and the target tool's schema to semantically
        map outputs to inputs (e.g. governing_class → section_class).
        """
        context_lines: list[str] = []
        for src_tool, result in all_tool_outputs.items():
            outputs = result.get("outputs", {})
            if outputs:
                context_lines.append(f"  {src_tool}: {json.dumps(outputs)}")
        if not context_lines:
            return {}

        params_info: dict[str, Any] = {}
        for p in params_to_resolve:
            info: dict[str, Any] = {
                k: v
                for k, v in schema_props.get(p, {}).items()
                if k in ("type", "description", "minimum", "maximum", "default")
            }
            cur = current_inputs.get(p)
            if cur is not None:
                info["current_value"] = cur
            params_info[p] = info

        tool_desc = ""
        entry = self.tool_registry.get(tool_name)
        if entry:
            tool_desc = entry.description

        prompt = (
            "###TASK:RESOLVE_INPUTS###\n"
            f"Tool: {tool_name}\n"
            f"Description: {tool_desc}\n\n"
            f"Parameters to resolve:\n{json.dumps(params_info, indent=2)}\n\n"
            f"Known inputs: {json.dumps({k: v for k, v in current_inputs.items() if v is not None})}\n\n"
            "Previous tool executions and their outputs:\n"
            + "\n".join(context_lines) + "\n\n"
        )
        if plan_context:
            prompt += f"Execution plan:\n{json.dumps(plan_context, indent=2)}\n\n"
        prompt += (
            "For each parameter, determine the correct value from the prior "
            "tool outputs. Use SEMANTIC understanding — output names from prior "
            "tools may differ from input parameter names (e.g., a classification "
            "tool's output class should map to a resistance tool's class input). "
            "Override current_value or schema defaults when a prior tool computed "
            "a relevant result. Return ONLY a JSON object of "
            "{param_name: value}. Omit any that cannot be determined."
        )
        try:
            raw = self.orchestrator_llm.generate(
                system_prompt=(
                    "You resolve tool input parameters from available "
                    "session data. Use semantic understanding to map outputs "
                    "from prior tools to input parameters, even when names "
                    "differ. Return only valid JSON."
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

    # ── Intent classification ──────────────────────────────────────

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

        cleaned = self._ATTACHMENT_MARKER_RE.sub("", query).strip()
        lowered = cleaned.lower()
        words = lowered.split()
        has_eng = any(kw in lowered for kw in self._ENG_KEYWORDS)

        calc_verbs = {"calculate", "check", "verify", "design", "determine", "compute", "find"}
        if has_eng and any(v in lowered for v in calc_verbs):
            return {"intent": "pipeline"}

        if len(words) <= 4 and lowered.rstrip("!.,?") in self._GREETINGS:
            return {"intent": "greeting"}

        if has_eng and not has_images:
            return {"intent": "pipeline"}

        llm_intent = self._llm_classify(cleaned, attachments, has_images)
        if llm_intent:
            return {"intent": llm_intent}

        if has_images:
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

            logger.warning("intent_classification_unexpected", extra={"raw": raw.strip()})
            return None

        except Exception as exc:
            logger.warning("intent_classification_failed", extra={"error": str(exc)})
            return None

    # ── Direct response handlers ───────────────────────────────────

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
            yield ("thinking", {"content": "Off-topic query — responding directly."})
            yield ("machine", {
                "node": "intake", "status": "done",
                "title": "Query Intake", "detail": "Off-topic query — quick response.",
            })
        elif intent == "greeting":
            yield ("thinking", {"content": "Greeting received."})
            yield ("machine", {
                "node": "intake", "status": "done",
                "title": "Query Intake", "detail": "Greeting received.",
            })
        else:
            yield ("thinking", {"content": "Answering directly — no database or tools needed."})
            yield ("machine", {
                "node": "intake", "status": "done",
                "title": "Query Intake", "detail": "Answering directly — no database or tools needed.",
            })

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

        if intent == "decline":
            return self._DECLINE_ANSWER

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
                cleaned = raw.rstrip().rstrip(",").rstrip()
                if not cleaned or len(cleaned) < 10:
                    return _GREETING_FALLBACK
                _BROKEN_TAILS = {
                    "and", "or", "but", "the", "a", "an", "to", "for",
                    "with", "in", "on", "at", "of", "is", "am", "are",
                    "i", "we", "you", "that", "this", "my", "your",
                }
                last_word = cleaned.rstrip(".!?,;:").rsplit(None, 1)[-1].lower()
                if last_word in _BROKEN_TAILS:
                    logger.warning("greeting_truncated", extra={"raw": raw[:120]})
                    return _GREETING_FALLBACK
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

    # ── Follow-up resolution ───────────────────────────────────────

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

    # ── Source collection ──────────────────────────────────────────

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

    # ── Formatting utilities ───────────────────────────────────────

    @staticmethod
    def _pretty_tool_name(tool_name: str) -> str:
        pretty = tool_name.replace("_ec3", "").replace("_", " ").title()
        pretty = pretty.replace("Ipe", "IPE").replace("Ec3", "EC3")
        return pretty

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

    @staticmethod
    def _format_clause_text_for_display(text: str) -> str:
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
