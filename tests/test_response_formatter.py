from __future__ import annotations

from backend.tools.response_formatter import ResponseFormatterTool


def test_polish_narrative_fixes_broken_tail_and_adds_basis() -> None:
    formatter = ResponseFormatterTool()

    polished = formatter.polish_narrative(
        "The design shear resistance V_c,Rd for the IPE400 cannot be calculated because the specific cross",
        basis="This follows EN 1993-1-1, Cl. 6.2.10 (Bending, shear and axial force).",
    )

    assert "because required section properties are missing." in polished
    assert "This follows EN 1993-1-1, Cl. 6.2.10" in polished


def test_format_markdown_applies_subscripts() -> None:
    formatter = ResponseFormatterTool()
    assert formatter.format_markdown("M_Rd with gamma_M0 and f_y") == (
        "M<sub>Rd</sub> with γ<sub>M0</sub> and f<sub>y</sub>"
    )


def test_pretty_key_and_value_formatting() -> None:
    formatter = ResponseFormatterTool()
    assert formatter.pretty_key("M_Rd_kNm") == "M<sub>Rd</sub> (kNm)"
    assert formatter.format_value("M_Rd_kNm", 123.456) == "123.46 kNm"
