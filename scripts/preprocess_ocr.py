"""Pre-process OCR JSON files into structured clauses and annex entries."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

NUMERIC_HEADING_RE = re.compile(
    r"^(?P<cid>\d{1,2}(?:\.\d{1,3}){0,5}(?:\(\d+\))?)\s+(?P<title>[^\n]{3,180})$"
)
ANNEX_CLAUSE_RE = re.compile(
    r"^(?P<cid>[A-Z]{1,2}\.\d+(?:\.\d+){0,5}(?:\(\d+\))?)\s+(?P<title>[^\n]{3,180})$"
)
TABLE_HEADING_RE = re.compile(
    r"^Table\s+(?P<num>[A-Z]{0,2}\.?\d+(?:\.\d+)*)\s*[:\-]?\s*(?P<title>[^\n]*)$",
    re.IGNORECASE,
)
ANNEX_HEADING_RE = re.compile(
    r"^(?P<cid>Annex\s+[A-Z]{1,2})"
    r"(?:\s*[\[(][^\])]{0,60}[\])])?"
    r"\s*(?:[-]\s*)?"
    r"(?P<title>[^\n].*)$",
    re.IGNORECASE,
)
ANNEX_STANDALONE_RE = re.compile(
    r"^(?P<cid>Annex\s+[A-Z]{1,2})(?:\s*[\[(][^\])]{0,60}[\])])?$",
    re.IGNORECASE,
)
HEADING_LINE_RE = re.compile(
    r"^(?:Annex\s+[A-Z]{1,2}|Table\s+[A-Z]{0,2}\.?\d+(?:\.\d+)*|"
    r"[A-Z]{1,2}\.\d+(?:\.\d+){0,5}(?:\(\d+\))?|\d{1,2}(?:\.\d{1,3}){0,5}(?:\(\d+\))?)\b",
    re.IGNORECASE,
)

SKIP_TITLE_PREFIXES = ("the ", "this ", "for ", "in ", "a ", "an ", "where ", "see ", "if ")


@dataclass
class Heading:
    pos: int
    clause_id: str
    title: str
    drop_lines: int


def load_ocr(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pages: list[str] = []

    if isinstance(raw, dict):
        for key in ("pages", "items", "content"):
            value = raw.get(key)
            if isinstance(value, list):
                raw = value
                break
        else:
            raw = [raw]

    if not isinstance(raw, list):
        return pages

    for item in raw:
        if isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict) and sub.get("content"):
                    pages.append(str(sub["content"]))
            continue
        if isinstance(item, dict) and item.get("content"):
            pages.append(str(item["content"]))

    return pages


def extract_clauses(pages: list[str], *, pointer_prefix: str) -> list[dict]:
    full_text = "\n\n".join(pages)
    headings = _extract_headings(full_text)
    if not headings:
        return []

    best_by_id: dict[str, tuple[int, str, str, int]] = {}
    for idx, heading in enumerate(headings):
        end = headings[idx + 1].pos if idx + 1 < len(headings) else len(full_text)
        segment = full_text[heading.pos:end].strip()
        body = _extract_body(segment, heading.drop_lines)
        if len(body) > 3000:
            body = body[:3000] + "..."
        body_len = len(body)
        min_len = 10 if heading.clause_id.startswith(("Annex ", "Table ")) else 20
        if body_len < min_len:
            continue

        prev = best_by_id.get(heading.clause_id)
        # Prefer later heading occurrences to avoid selecting table-of-contents entries.
        if prev is None or heading.pos > prev[0] or (heading.pos == prev[0] and body_len > prev[3]):
            best_by_id[heading.clause_id] = (heading.pos, heading.title, body, body_len)

    clauses: list[dict] = []
    for clause_id, (pos, title, body, _) in sorted(best_by_id.items(), key=lambda x: x[1][0]):
        clauses.append(
            {
                "clause_id": clause_id,
                "title": title,
                "text": body,
                "pointer": f"{pointer_prefix}#/{clause_id}",
                "keywords": _extract_keywords(clause_id, title, body),
                "_pos": pos,
            }
        )

    # Keep pointer order stable but remove helper field.
    for clause in clauses:
        clause.pop("_pos", None)

    return clauses


def _extract_headings(full_text: str) -> list[Heading]:
    raw_lines = full_text.splitlines(keepends=True)
    lines = [line.rstrip("\r\n") for line in raw_lines]

    headings: list[Heading] = []
    pos = 0
    for idx, raw_line in enumerate(lines):
        parsed = _parse_heading(lines, idx)
        if parsed is not None:
            clause_id, title, drop_lines = parsed
            headings.append(Heading(pos=pos, clause_id=clause_id, title=title, drop_lines=drop_lines))
        pos += len(raw_lines[idx])
    return headings


def _parse_heading(lines: list[str], idx: int) -> tuple[str, str, int] | None:
    line = _clean_heading_line(lines[idx])
    if not line:
        return None

    annex_match = ANNEX_HEADING_RE.match(line)
    if annex_match:
        clause_id = _normalize_annex_id(annex_match.group("cid"))
        title = _clean_title(annex_match.group("title"))
        if _is_valid_title(title, allow_prefix=True):
            return clause_id, title, 1

    annex_standalone = ANNEX_STANDALONE_RE.match(line)
    if annex_standalone:
        clause_id = _normalize_annex_id(annex_standalone.group("cid"))
        title, consumed = _read_following_title(lines, idx + 1)
        if title is None:
            title = clause_id
            consumed = 0
        return clause_id, title, 1 + consumed

    table_match = TABLE_HEADING_RE.match(line)
    if table_match:
        table_no = table_match.group("num").upper()
        clause_id = f"Table {table_no}"
        title = _clean_title(table_match.group("title")) or clause_id
        if _is_valid_title(title, allow_prefix=True):
            return clause_id, title, 1

    annex_clause_match = ANNEX_CLAUSE_RE.match(line)
    if annex_clause_match:
        clause_id = annex_clause_match.group("cid")
        title = _clean_title(annex_clause_match.group("title"))
        if _is_valid_title(title):
            return clause_id, title, 1

    numeric_match = NUMERIC_HEADING_RE.match(line)
    if numeric_match:
        clause_id = numeric_match.group("cid")
        title = _clean_title(numeric_match.group("title"))
        if _is_valid_title(title):
            return clause_id, title, 1

    return None


def _read_following_title(lines: list[str], start_idx: int) -> tuple[str | None, int]:
    for offset, line in enumerate(lines[start_idx : start_idx + 4], start=1):
        cleaned = _clean_heading_line(line)
        if not cleaned:
            continue
        if HEADING_LINE_RE.match(cleaned):
            return None, 0
        title = _clean_title(cleaned)
        if not re.match(r"^[A-Za-z]", title):
            return None, 0
        if _is_valid_title(title, allow_prefix=True):
            return title, offset
        return None, 0
    return None, 0


def _clean_heading_line(line: str) -> str:
    cleaned = line.replace("\u2013", "-").replace("\u2014", "-").strip()
    cleaned = re.sub(r"^[#>*\-\s]+", "", cleaned)
    cleaned = cleaned.strip("*").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _clean_title(title: str) -> str:
    cleaned = title.strip(" .:-")
    cleaned = re.sub(r"\s*\.{2,}\s*\d+\s*$", "", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z\)])\s+\d{1,3}$", "", cleaned)
    cleaned = cleaned.strip(" .:-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _normalize_annex_id(raw_annex_id: str) -> str:
    suffix = raw_annex_id.strip().split()[-1].upper()
    return f"Annex {suffix}"


def _is_valid_title(title: str, *, allow_prefix: bool = False) -> bool:
    if len(title) < 3 or len(title) > 180:
        return False
    if "..." in title:
        return False
    if not allow_prefix and title.lower().startswith(SKIP_TITLE_PREFIXES):
        return False
    if re.fullmatch(r"[\d.]+", title):
        return False
    return True


def _extract_body(segment: str, drop_lines: int) -> str:
    lines = segment.splitlines()
    body_lines = lines[drop_lines:] if len(lines) > drop_lines else []
    body = "\n".join(body_lines).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


def _extract_keywords(clause_id: str, title: str, text: str) -> list[str]:
    kw: set[str] = set()
    kw.add(clause_id)
    for word in title.lower().split():
        token = word.strip(".,:;()[]{}")
        if len(token) > 3 and token.isalpha():
            kw.add(token)

    eng_terms = [
        "bending",
        "shear",
        "axial",
        "compression",
        "tension",
        "buckling",
        "resistance",
        "moment",
        "force",
        "deflection",
        "classification",
        "yield",
        "ultimate",
        "weld",
        "bolt",
        "connection",
        "section",
        "imperfection",
        "stability",
        "lateral",
        "torsional",
        "cross-section",
        "elastic",
        "plastic",
        "modulus",
        "flange",
        "web",
        "slenderness",
        "serviceability",
        "fatigue",
        "fracture",
        "ductility",
        "annex",
        "table",
        "fire",
        "stainless",
        "shell",
    ]
    text_lower = text.lower()
    for term in eng_terms:
        if term in text_lower:
            kw.add(term)

    return sorted(kw)


def _structured_name_for(source_path: Path) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", source_path.stem.lower()).strip("_")
    return f"{base}_structured.json"


def process_sources(source_files: Iterable[Path], data_dir: Path) -> list[tuple[Path, int, int]]:
    results: list[tuple[Path, int, int]] = []
    for source in sorted(source_files):
        pages = load_ocr(source)
        pointer_prefix = re.sub(r"[^a-z0-9]+", "_", source.stem.lower()).strip("_")
        clauses = extract_clauses(pages, pointer_prefix=pointer_prefix)

        out_path = data_dir / _structured_name_for(source)
        out_payload = {"source_file": source.name, "clauses": clauses}
        out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        annex_count = sum(1 for c in clauses if c["clause_id"].startswith("Annex "))
        results.append((out_path, len(clauses), annex_count))
    return results


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "ec3"
    source_files = list(data_dir.glob("EN 1993-*.json"))

    if not source_files:
        print(f"No source files found in {data_dir}")
        return

    print(f"Processing {len(source_files)} source files from {data_dir}")
    results = process_sources(source_files, data_dir)
    print("")
    for out_path, clause_count, annex_count in results:
        print(f"- {out_path.name}: {clause_count} clauses ({annex_count} annex headings)")


if __name__ == "__main__":
    main()
