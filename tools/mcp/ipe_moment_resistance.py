from __future__ import annotations

from pydantic import BaseModel, Field, PositiveFloat, field_validator

from tools.mcp.cli import run_cli
from tools.mcp.ipe_section_library import load_ipe_sections
from tools.mcp.section_library import steel_grade_to_fy

TOOL_NAME = "ipe_moment_resistance_ec3"


class IPEMomentResistanceInput(BaseModel):
    section_name: str = Field(description="IPE section name, e.g., IPE300")
    steel_grade: str = "S355"
    fy_mpa: PositiveFloat | None = None
    section_class: int = Field(default=2, ge=1, le=4)
    gamma_M0: PositiveFloat = Field(default=1.0)

    @field_validator("section_name")
    @classmethod
    def validate_section_name(cls, value: str) -> str:
        normalized = value.strip().upper().replace(" ", "")
        if not normalized.startswith("IPE"):
            raise ValueError("section_name must be an IPE section, e.g., IPE300.")
        return normalized

    @field_validator("steel_grade")
    @classmethod
    def validate_steel_grade(cls, value: str) -> str:
        value = value.strip().upper()
        if not (value.startswith("S") and value[1:].isdigit()):
            raise ValueError("steel_grade must look like S355.")
        return value


def compute_ipe_moment_resistance(input_data: IPEMomentResistanceInput) -> dict:
    sections, source = load_ipe_sections()
    section_name = input_data.section_name
    row = sections.get(section_name)
    if row is None:
        known = ", ".join(sorted(list(sections.keys())[:12]))
        raise ValueError(
            f"Section '{section_name}' not found in IPE library source '{source}'. "
            f"Known examples: {known}."
        )

    fy = float(input_data.fy_mpa) if input_data.fy_mpa is not None else steel_grade_to_fy(input_data.steel_grade)
    gamma = float(input_data.gamma_M0)
    section_class = int(input_data.section_class)

    w_pl = float(row["wpl_y_cm3"])
    w_el = float(row["wel_y_cm3"])

    # EC3 EN 1993-1-1, Clause 6.2.5:
    #   Class 1-2: plastic modulus Wpl
    #   Class 3:   elastic modulus Wel
    #   Class 4:   effective section modulus Weff (requires EN 1993-1-5)
    #              — use Wel as conservative approximation with warning
    if section_class <= 2:
        w_used_cm3 = w_pl
        resistance_basis = "Plastic modulus (Wpl) for Class 1-2."
        class_note = "Plastic resistance basis"
    elif section_class == 3:
        w_used_cm3 = w_el
        resistance_basis = "Elastic modulus (Wel) for Class 3."
        class_note = "Elastic resistance basis"
    else:  # Class 4
        w_used_cm3 = w_el
        resistance_basis = (
            "Class 4: effective section modulus (Weff) required per EN 1993-1-5. "
            "Using elastic modulus (Wel) as conservative approximation. "
            "Actual Weff may be lower due to local buckling of the web."
        )
        class_note = "Elastic modulus approximation (Class 4 — Weff not computed)"

    w_used_mm3 = w_used_cm3 * 1000.0
    m_rd_knm = (w_used_mm3 * fy / gamma) / 1_000_000.0

    return {
        "inputs_used": {
            "section_name": section_name,
            "steel_grade": input_data.steel_grade,
            "fy_mpa": round(fy, 3),
            "section_class": section_class,
            "gamma_M0": gamma,
        },
        "section_properties": {
            "h_mm": float(row["h_mm"]),
            "b_mm": float(row["b_mm"]),
            "tw_mm": float(row["tw_mm"]),
            "tf_mm": float(row["tf_mm"]),
            "area_cm2": float(row["area_cm2"]),
            "wel_y_cm3": float(row["wel_y_cm3"]),
            "wpl_y_cm3": float(row["wpl_y_cm3"]),
            "library_source": source,
        },
        "intermediate": {
            "w_used_cm3": round(w_used_cm3, 3),
            "w_used_mm3": round(w_used_mm3, 3),
            "resistance_basis": resistance_basis,
        },
        "outputs": {
            "M_Rd_kNm": round(m_rd_knm, 3),
            "formula": "M_Rd = W * f_y / gamma_M0",
            "section_class_interpretation": class_note,
        },
        "clause_references": [
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.2.5(1)",
                "title": "Bending resistance",
                "pointer": "en_1993_1_1_sample.json#/clauses/4",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "5.5.2(1)",
                "title": "Classification of cross-sections",
                "pointer": "en_1993_1_1_sample.json#/clauses/1",
            },
        ],
        "notes": [
            "This tool focuses on cross-section moment resistance only (no buckling/LTB checks).",
            f"IPE profile source used: {source}.",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=IPEMomentResistanceInput, handler=compute_ipe_moment_resistance)
