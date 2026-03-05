from __future__ import annotations

import math

from pydantic import BaseModel, Field, PositiveFloat, field_validator, model_validator

from tools.mcp.cli import run_cli
from tools.mcp.section_library import SECTION_LIBRARY, steel_grade_to_fy

TOOL_NAME = "member_resistance_ec3"


class MemberResistanceInput(BaseModel):
    section_name: str | None = None
    steel_grade: str = "S355"
    fy_mpa: PositiveFloat | None = None

    section_class: int = Field(default=2, ge=1, le=4)
    gamma_M0: PositiveFloat = Field(default=1.0)

    area_cm2: PositiveFloat | None = None
    wpl_y_cm3: PositiveFloat | None = None
    wel_y_cm3: PositiveFloat | None = None
    av_z_cm2: PositiveFloat | None = None

    @field_validator("steel_grade")
    @classmethod
    def validate_steel_grade(cls, value: str) -> str:
        value = value.strip().upper()
        if not (value.startswith("S") and value[1:].isdigit()):
            raise ValueError("steel_grade must look like S355.")
        return value

    @model_validator(mode="after")
    def fill_from_library(self) -> "MemberResistanceInput":
        if self.section_name:
            key = self.section_name.upper().replace(" ", "")
            if key in SECTION_LIBRARY:
                row = SECTION_LIBRARY[key]
                if self.area_cm2 is None:
                    self.area_cm2 = float(row["area_cm2"])
                if self.wpl_y_cm3 is None:
                    self.wpl_y_cm3 = float(row["wpl_y_cm3"])
                if self.wel_y_cm3 is None:
                    self.wel_y_cm3 = float(row["wel_y_cm3"])
                if self.av_z_cm2 is None:
                    self.av_z_cm2 = float(row["av_z_cm2"])

        if self.fy_mpa is None:
            self.fy_mpa = steel_grade_to_fy(self.steel_grade)

        missing = [
            name
            for name in ["area_cm2", "wpl_y_cm3", "wel_y_cm3", "av_z_cm2"]
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "Missing section properties: "
                + ", ".join(missing)
                + ". Provide section_name with known section or explicit properties in cm2/cm3."
            )

        return self


def compute_resistance(input_data: MemberResistanceInput) -> dict:
    fy = float(input_data.fy_mpa)
    gamma = float(input_data.gamma_M0)
    section_class = int(input_data.section_class)

    # EC3 EN 1993-1-1, Clause 6.2.5:
    #   Class 1-2: plastic modulus Wpl
    #   Class 3:   elastic modulus Wel
    #   Class 4:   effective section modulus Weff (use Wel as approximation)
    if section_class <= 2:
        w_used_cm3 = float(input_data.wpl_y_cm3)
        bending_basis = "M_Rd = Wpl * f_y / gamma_M0"
    else:
        w_used_cm3 = float(input_data.wel_y_cm3)
        bending_basis = (
            "M_Rd = Wel * f_y / gamma_M0"
            if section_class == 3
            else "M_Rd = Wel * f_y / gamma_M0 (Class 4 approx — Weff required per EN 1993-1-5)"
        )

    w_used_mm3 = w_used_cm3 * 1000.0
    area_mm2 = float(input_data.area_cm2) * 100.0
    av_mm2 = float(input_data.av_z_cm2) * 100.0

    mrd_knm = (w_used_mm3 * fy / gamma) / 1_000_000.0
    nrd_kn = (area_mm2 * fy / gamma) / 1_000.0
    vrd_kn = (av_mm2 * fy / (math.sqrt(3) * gamma)) / 1_000.0

    return {
        "inputs_used": {
            "section_name": input_data.section_name,
            "steel_grade": input_data.steel_grade,
            "fy_mpa": round(fy, 3),
            "section_class": section_class,
            "gamma_M0": gamma,
            "w_used_cm3": round(w_used_cm3, 3),
            "area_cm2": float(input_data.area_cm2),
            "av_z_cm2": float(input_data.av_z_cm2),
        },
        "intermediate": {
            "w_used_mm3": round(w_used_mm3, 3),
            "area_mm2": round(area_mm2, 3),
            "av_mm2": round(av_mm2, 3),
        },
        "outputs": {
            "M_Rd_kNm": round(mrd_knm, 3),
            "N_Rd_kN": round(nrd_kn, 3),
            "V_Rd_kN": round(vrd_kn, 3),
            "bending_formula_basis": bending_basis,
            "axial_formula_basis": "N_Rd = A * f_y / gamma_M0",
            "shear_formula_basis": "V_Rd = A_v * f_y / (sqrt(3) * gamma_M0)",
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
                "clause_id": "6.2.4(1)",
                "title": "Compression resistance",
                "pointer": "en_1993_1_1_sample.json#/clauses/3",
            },
            {
                "doc_id": "ec3.en1993-1-1.2005",
                "clause_id": "6.2.6(1)",
                "title": "Shear resistance",
                "pointer": "en_1993_1_1_sample.json#/clauses/5",
            },
        ],
        "notes": [
            "Member buckling and lateral torsional buckling are outside this MVP placeholder calculator.",
        ],
    }


if __name__ == "__main__":
    run_cli(tool_name=TOOL_NAME, input_model=MemberResistanceInput, handler=compute_resistance)
