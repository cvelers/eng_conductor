from __future__ import annotations

import re
from typing import Any

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

_LATEX_BLOCK_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^\$\n]+?\$)")

_DANGLING_TAIL_RE = re.compile(
    r"(?:\b(?:and|or|with|because|for|to|of|in|on|at|from|that|which|where|when|if|while|as)\b[\s,;:]*)+$",
    re.IGNORECASE,
)

_BROKEN_HYPHEN_TAIL_RE = re.compile(r"(?:\b[0-9A-Za-z_]+-)$")
_INCOMPLETE_CAUSE_TAIL_RE = re.compile(
    r"(?:\b(?:because|due to|as)\b[^.?!]{0,160}?\b(?:cross|cross[- ]?section|section|properties?|inputs?|data|parameters?)\b)$",
    re.IGNORECASE,
)


class ResponseFormatterTool:
    def polish_narrative(
        self,
        narrative: str,
        *,
        headline: str = "",
        basis: str = "",
    ) -> str:
        # Normalize horizontal whitespace within lines but preserve line breaks
        text = re.sub(r"[^\S\n]+", " ", (narrative or "")).strip()
        if not text:
            text = headline if headline else "Results computed from the available tools and EC3 database."
            text = self._ensure_sentence_complete(text)
            text = self._sanitize_bold_markers(text)
        else:
            text = self._ensure_sentence_complete(text)
            text = self._sanitize_bold_markers(text)
            # Do NOT override the LLM's first sentence with a generated headline.
            # The LLM prompt already produces a well-structured opening sentence.

        if basis and "Cl." not in text:
            text = f"{text} {basis}".strip()

        return self._ensure_sentence_complete(text)

    def format_markdown(self, text: str) -> str:
        formatted = text or ""
        # Split on LaTeX blocks so regex substitutions don't corrupt formulas
        parts = _LATEX_BLOCK_RE.split(formatted)
        result = []
        for part in parts:
            if _LATEX_BLOCK_RE.fullmatch(part):
                result.append(part)  # LaTeX — preserve as-is
            else:
                for pattern, replacement in _NARRATIVE_SUB_RE:
                    part = pattern.sub(replacement, part)
                result.append(part)
        return "".join(result)

    def strip_unit_suffix(self, key: str) -> str:
        for suffix, _ in _UNIT_SUFFIXES:
            if key.endswith(suffix):
                return key[: -len(suffix)]
        return key

    def format_value(self, key: str, val: Any) -> str:
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

    def pretty_key(self, key: str) -> str:
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

    @staticmethod
    def _guess_unit(key: str) -> str:
        key_lower = key.lower()
        if "knm" in key_lower:
            return "kNm"
        if "kn" in key_lower:
            return "kN"
        if "mpa" in key_lower:
            return "MPa"
        if "gpa" in key_lower:
            return "GPa"
        if "_mm" in key_lower:
            return "mm"
        if key_lower.endswith("_m"):
            return "m"
        if "cm2" in key_lower:
            return "cm²"
        if "cm3" in key_lower:
            return "cm³"
        if "cm4" in key_lower:
            return "cm⁴"
        return ""
