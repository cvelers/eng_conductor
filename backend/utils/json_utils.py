from __future__ import annotations

import json
import re
from typing import Any

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        first_nl = value.find("\n")
        if first_nl >= 0:
            value = value[first_nl + 1 :]
    if value.endswith("```"):
        value = value[:-3]
    return value.strip()


def parse_json_loose(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty input")

    candidates: list[str] = [raw, strip_code_fences(raw)]
    for match in _FENCED_JSON_RE.finditer(raw):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue

    extracted = _extract_first_balanced_json(raw)
    if extracted:
        return json.loads(extracted)

    raise ValueError("could not parse JSON payload")


def _extract_first_balanced_json(text: str) -> str:
    open_candidates = []
    for ch in ("{", "["):
        idx = text.find(ch)
        if idx >= 0:
            open_candidates.append((idx, ch))
    if not open_candidates:
        return ""

    start_idx, start_char = min(open_candidates, key=lambda item: item[0])
    end_char = "}" if start_char == "{" else "]"

    depth = 0
    in_string = False
    escaped = False

    for pos in range(start_idx, len(text)):
        ch = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == start_char:
            depth += 1
            continue
        if ch == end_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : pos + 1]

    return ""
