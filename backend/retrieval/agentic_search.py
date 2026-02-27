from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator

from backend.config import Settings
from backend.llm.base import LLMProvider
from backend.registries.document_registry import ClauseRecord
from backend.utils.json_utils import parse_json_loose, strip_code_fences

logger = logging.getLogger(__name__)

QUERY_SANITIZE_RE = re.compile(r"[^A-Za-z0-9\s_\-\./]")
TOKEN_RE = re.compile(r"[a-z0-9_\-\.]+")
CLAUSE_ID_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+)?(?:\([^)]+\))?)\b")

STOPWORDS = frozenset({
    "the", "is", "in", "of", "to", "and", "for", "with", "that", "this",
    "are", "from", "be", "as", "by", "on", "it", "an", "or", "at", "if",
    "shall", "may", "should", "can", "which", "where", "when", "how",
    "what", "does", "has", "been", "its", "not", "but", "than", "into",
})


@dataclass
class RetrievedClause:
    clause: ClauseRecord
    score: float
    matched_terms: list[str]


@dataclass
class _IndexedClause:
    """Pre-processed clause data for fast search."""
    clause: ClauseRecord
    title_tokens: set[str]
    text_tokens: set[str]
    keyword_tokens: set[str]
    all_tokens: set[str]
    full_text_lower: str


class AgenticRetriever:
    def __init__(
        self,
        *,
        settings: Settings,
        search_provider: LLMProvider,
        clauses: list[ClauseRecord],
    ) -> None:
        self.settings = settings
        self.search_provider = search_provider
        self.clauses = clauses
        self._entries: list[_IndexedClause] = []
        self._inverted_index: dict[str, set[int]] = defaultdict(set)
        self._doc_freq: dict[str, int] = {}
        self._total_docs = max(len(clauses), 1)
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        for idx, clause in enumerate(self.clauses):
            title_tokens = self._tokenize(clause.clause_title)
            text_tokens = self._tokenize(clause.text)
            keyword_tokens = self._tokenize(" ".join(clause.keywords))
            all_tokens = title_tokens | text_tokens | keyword_tokens

            full_text = " ".join([
                clause.standard, clause.clause_id, clause.clause_title,
                clause.text, " ".join(clause.keywords),
            ]).lower()

            entry = _IndexedClause(
                clause=clause,
                title_tokens=title_tokens,
                text_tokens=text_tokens,
                keyword_tokens=keyword_tokens,
                all_tokens=all_tokens,
                full_text_lower=full_text,
            )
            self._entries.append(entry)

            for token in all_tokens:
                self._inverted_index[token].add(idx)

        self._doc_freq = {
            token: len(indices) for token, indices in self._inverted_index.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int | None = None,
    ) -> tuple[list[RetrievedClause], list[dict[str, object]]]:
        final_results: list[RetrievedClause] = []
        trace: list[dict[str, object]] = []
        for event in self.iter_retrieve(query, top_k=top_k):
            if event.get("type") == "final":
                final_results = event.get("results", [])
                trace = event.get("trace", [])
        return final_results, trace

    def iter_retrieve(
        self, query: str, top_k: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        limit = top_k or self.settings.top_k_clauses
        trace: list[dict[str, object]] = []
        aggregated: dict[str, RetrievedClause] = {}

        agentic = self.settings.agentic_search_enabled and self.search_provider.available
        max_gap_iters = max(0, self.settings.max_retrieval_iters - 2) if agentic else 0

        # ---- Phase 1: query decomposition ----
        sub_queries = self._decompose_query(query) if agentic else [self._sanitize(query)]

        trace.append({
            "iteration": 1, "query": query,
            "top_clause_ids": [], "phase": "decompose",
            "sub_queries": sub_queries,
        })

        # ---- Phase 2: lexical candidate generation ----
        candidate_pool: dict[str, RetrievedClause] = {}
        for sq in sub_queries:
            for hit in self._search_lexical(sq, limit=limit * 4):
                key = hit.clause.citation_address
                if key not in candidate_pool or hit.score > candidate_pool[key].score:
                    candidate_pool[key] = hit

        candidates = sorted(candidate_pool.values(), key=lambda x: -x.score)[: limit * 3]

        step_lexical = {
            "iteration": 1, "phase": "lexical", "query": query,
            "top_clause_ids": [
                f"{c.clause.doc_id}:{c.clause.clause_id}" for c in candidates[:limit]
            ],
        }
        trace.append(step_lexical)

        logger.info(
            "retrieval_lexical",
            extra={"hits": len(candidates), "top": [c.clause.clause_id for c in candidates[:3]]},
        )
        yield {
            "type": "iteration", "step": step_lexical,
            "top": [
                {"doc_id": c.clause.doc_id, "clause_id": c.clause.clause_id,
                 "title": c.clause.clause_title, "score": c.score}
                for c in candidates[:3]
            ],
        }

        # ---- Phase 3: LLM relevance re-ranking ----
        if agentic and candidates:
            scored = self._llm_score_relevance(query, candidates[:30])
            for item in scored:
                key = item.clause.citation_address
                if key not in aggregated or item.score > aggregated[key].score:
                    aggregated[key] = item

            step_rerank = {
                "iteration": 2, "phase": "rerank", "query": query,
                "top_clause_ids": [
                    f"{c.clause.doc_id}:{c.clause.clause_id}"
                    for c in sorted(aggregated.values(), key=lambda x: -x.score)[:limit]
                ],
            }
            trace.append(step_rerank)

            logger.info(
                "retrieval_reranked",
                extra={"hits": len(aggregated), "top": [
                    c.clause.clause_id
                    for c in sorted(aggregated.values(), key=lambda x: -x.score)[:3]
                ]},
            )
            yield {
                "type": "iteration", "step": step_rerank,
                "top": [
                    {"doc_id": c.clause.doc_id, "clause_id": c.clause.clause_id,
                     "title": c.clause.clause_title, "score": c.score}
                    for c in sorted(aggregated.values(), key=lambda x: -x.score)[:3]
                ],
            }
        else:
            for item in candidates[:limit]:
                aggregated[item.clause.citation_address] = item

        # ---- Phase 4: iterative gap analysis ----
        seen_gap_queries: set[str] = set()

        for gap_iter in range(max_gap_iters):
            current_top = sorted(aggregated.values(), key=lambda x: -x.score)[:limit]
            gap_query = self._llm_gap_analysis(query, current_top)
            if not gap_query or gap_query in seen_gap_queries:
                break
            seen_gap_queries.add(gap_query)

            gap_hits = self._search_lexical(gap_query, limit=limit * 2)
            new_candidates = [
                h for h in gap_hits if h.clause.citation_address not in aggregated
            ][:15]
            if not new_candidates:
                break

            if self.search_provider.available:
                scored_new = self._llm_score_relevance(query, new_candidates)
                for item in scored_new:
                    key = item.clause.citation_address
                    if key not in aggregated or item.score > aggregated[key].score:
                        aggregated[key] = item
            else:
                for item in new_candidates:
                    aggregated[item.clause.citation_address] = item

            step_gap = {
                "iteration": 3 + gap_iter, "phase": "gap_fill",
                "query": gap_query,
                "top_clause_ids": [
                    f"{c.clause.doc_id}:{c.clause.clause_id}"
                    for c in sorted(aggregated.values(), key=lambda x: -x.score)[:limit]
                ],
            }
            trace.append(step_gap)

            logger.info("retrieval_gap_fill", extra={"gap_query": gap_query})
            yield {
                "type": "iteration", "step": step_gap,
                "top": [
                    {"doc_id": c.clause.doc_id, "clause_id": c.clause.clause_id,
                     "title": c.clause.clause_title, "score": c.score}
                    for c in sorted(aggregated.values(), key=lambda x: -x.score)[:3]
                ],
            }

        # ---- Phase 5: recursive expansion (optional) ----
        merged = sorted(
            aggregated.values(),
            key=lambda item: (-item.score, item.clause.doc_id, item.clause.clause_id),
        )

        if self.settings.recursive_retrieval_enabled:
            merged = self._recursive_expand(merged, limit)
            yield {"type": "recursive", "detail": "Recursive retrieval expansion applied."}

        final_results = merged[:limit]
        yield {"type": "final", "results": final_results, "trace": trace}

    # ------------------------------------------------------------------
    # Lexical search with TF-IDF field-weighted scoring
    # ------------------------------------------------------------------

    def _sanitize(self, query: str) -> str:
        clean = QUERY_SANITIZE_RE.sub(" ", query).strip().lower()
        return re.sub(r"\s+", " ", clean)

    def _tokenize(self, value: str) -> set[str]:
        tokens = TOKEN_RE.findall(value.lower())
        return {t for t in tokens if len(t) > 1 and t not in STOPWORDS}

    def _idf(self, token: str) -> float:
        df = self._doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + self._total_docs / df)

    def _search_lexical(self, query: str, limit: int) -> list[RetrievedClause]:
        safe_query = self._sanitize(query)
        tokens = self._tokenize(safe_query)
        if not tokens:
            return []

        candidate_indices: dict[int, float] = {}
        for token in tokens:
            idf = self._idf(token)
            for idx in self._inverted_index.get(token, set()):
                candidate_indices[idx] = candidate_indices.get(idx, 0) + idf

        clause_id_patterns = set(CLAUSE_ID_RE.findall(safe_query))
        for idx, entry in enumerate(self._entries):
            cid = entry.clause.clause_id.lower()
            for qid in clause_id_patterns:
                if cid == qid or cid.startswith(qid):
                    candidate_indices[idx] = candidate_indices.get(idx, 0) + 5.0

        ranked: list[RetrievedClause] = []
        for idx in candidate_indices:
            entry = self._entries[idx]

            title_score = sum(self._idf(t) * 3.0 for t in tokens if t in entry.title_tokens)
            keyword_score = sum(self._idf(t) * 2.0 for t in tokens if t in entry.keyword_tokens)
            text_score = sum(self._idf(t) for t in tokens if t in entry.text_tokens)

            phrase_bonus = 4.0 if (len(safe_query) > 5 and safe_query in entry.full_text_lower) else 0.0

            matched = tokens & entry.all_tokens
            coverage_bonus = (len(matched) / len(tokens)) * 2.0 if tokens else 0.0

            total = title_score + keyword_score + text_score + phrase_bonus + coverage_bonus

            ranked.append(RetrievedClause(
                clause=entry.clause, score=total, matched_terms=sorted(matched),
            ))

        ranked.sort(key=lambda x: (-x.score, x.clause.doc_id, x.clause.clause_id))
        return ranked[:limit]

    # ------------------------------------------------------------------
    # LLM-powered search operations
    # ------------------------------------------------------------------

    def _decompose_query(self, query: str) -> list[str]:
        try:
            raw = self.search_provider.generate(
                system_prompt=(
                    "You decompose engineering queries into focused search sub-queries "
                    "for a Eurocode clause database.\n"
                    "Return a JSON array of 1-4 concise search strings. "
                    "Each should target a different aspect of the question.\n"
                    "Always include the original query (possibly simplified) as the first element."
                ),
                user_prompt=(
                    "###TASK:DECOMPOSE###\n"
                    f"Query: {query}\n\n"
                    "Return JSON array of search strings only."
                ),
                temperature=0,
                max_tokens=self.settings.search_decompose_max_tokens,
            )
            data = parse_json_loose(raw)
            if isinstance(data, list) and data:
                queries = [self._sanitize(str(q)) for q in data[:4] if str(q).strip()]
                if queries:
                    return queries
        except Exception as exc:
            logger.warning("query_decomposition_failed", extra={"error": str(exc)})

        return [self._sanitize(query)]

    def _llm_score_relevance(
        self, query: str, candidates: list[RetrievedClause],
    ) -> list[RetrievedClause]:
        if not candidates:
            return []
        if not self.search_provider.available:
            return candidates

        descriptions: list[str] = []
        for i, c in enumerate(candidates):
            snippet = c.clause.text[:250].replace("\n", " ")
            descriptions.append(
                f"{i + 1}. [{c.clause.clause_id}] {c.clause.clause_title}: {snippet}"
            )

        try:
            raw = self.search_provider.generate(
                system_prompt=(
                    "You are a Eurocode relevance scorer. "
                    "Score each clause for relevance to the engineering query.\n"
                    "Return a JSON array: [{\"idx\": 1, \"score\": 0-10}, ...]\n"
                    "10 = directly answers the query with key formulas/rules.\n"
                    "7-9 = highly relevant, contains needed information.\n"
                    "4-6 = related context.\n"
                    "0-3 = tangentially related or irrelevant."
                ),
                user_prompt=(
                    "###TASK:RELEVANCE###\n"
                    f"Query: {query}\n\n"
                    "Clauses:\n" + "\n".join(descriptions) + "\n\n"
                    "Score each clause. Return JSON array only."
                ),
                temperature=0,
                max_tokens=1200,
            )
            data = parse_json_loose(raw)
            if isinstance(data, list):
                score_map: dict[int, float] = {}
                for item in data:
                    if isinstance(item, dict):
                        idx = int(item.get("idx", 0)) - 1
                        score = float(item.get("score", 0))
                        if 0 <= idx < len(candidates):
                            score_map[idx] = score

                result: list[RetrievedClause] = []
                for i, c in enumerate(candidates):
                    llm_score = score_map.get(i, c.score)
                    result.append(RetrievedClause(
                        clause=c.clause,
                        score=llm_score,
                        matched_terms=c.matched_terms + ["llm_scored"],
                    ))
                result.sort(key=lambda x: (-x.score, x.clause.doc_id, x.clause.clause_id))
                return result
        except Exception as exc:
            logger.warning("llm_relevance_scoring_failed", extra={"error": str(exc)})

        return candidates

    def _llm_gap_analysis(
        self, query: str, current_results: list[RetrievedClause],
    ) -> str | None:
        if not current_results or not self.search_provider.available:
            return None

        summary = "\n".join(
            f"- [{c.clause.clause_id}] {c.clause.clause_title} (score: {c.score:.1f})"
            for c in current_results[:8]
        )

        try:
            raw = self.search_provider.generate(
                system_prompt=(
                    "You analyze Eurocode search results and identify gaps.\n"
                    "If important information is missing for answering the query, "
                    "return a JSON string with ONE focused search query to fill the gap.\n"
                    "If the results are sufficient, return JSON null."
                ),
                user_prompt=(
                    "###TASK:GAP###\n"
                    f"Query: {query}\n\n"
                    f"Current top results:\n{summary}\n\n"
                    "Return JSON: a search string or null."
                ),
                temperature=0,
                max_tokens=350,
            )
            data = parse_json_loose(raw)
            if isinstance(data, str) and data.strip():
                return self._sanitize(data)
        except Exception as exc:
            logger.warning("gap_analysis_failed", extra={"error": str(exc)})

        return None

    # ------------------------------------------------------------------
    # Recursive cross-reference expansion
    # ------------------------------------------------------------------

    def _recursive_expand(
        self, ranked: list[RetrievedClause], limit: int,
    ) -> list[RetrievedClause]:
        if not ranked:
            return ranked

        by_id = {f"{c.clause.doc_id}:{c.clause.clause_id}": c for c in ranked}
        clause_lookup = {
            f"{c.doc_id}:{c.clause_id}": c
            for c in [item.clause for item in ranked] + self.clauses
        }

        reference_re = re.compile(r"\b\d+\.\d+(?:\.\d+)?(?:\([^)]+\))?\b")
        for item in list(ranked):
            refs = reference_re.findall(item.clause.text)
            for ref in refs:
                for key, clause in clause_lookup.items():
                    if key in by_id:
                        continue
                    if clause.doc_id != item.clause.doc_id:
                        continue
                    if clause.clause_id == ref:
                        by_id[key] = RetrievedClause(
                            clause=clause,
                            score=max(0.1, item.score - 0.25),
                            matched_terms=["recursive_ref"],
                        )

        expanded = sorted(
            by_id.values(),
            key=lambda entry: (-entry.score, entry.clause.doc_id, entry.clause.clause_id),
        )
        return expanded[:limit]


def _strip_code_fences(text: str) -> str:
    # Backward-compat shim for existing call sites/tests.
    return strip_code_fences(text)
