from __future__ import annotations

from typing import Any

# Fallback sample values (used if eurocodepy is unavailable).
# Units: mm, cm2, cm3.
FALLBACK_IPE_SECTIONS: dict[str, dict[str, float]] = {
    "IPE200": {
        "h_mm": 200.0, "b_mm": 100.0, "tw_mm": 5.6, "tf_mm": 8.5,
        "area_cm2": 28.5, "I_y_cm4": 1943.0, "I_z_cm4": 142.0,
        "wel_y_cm3": 194.0, "wpl_y_cm3": 220.6,
    },
    "IPE240": {
        "h_mm": 240.0, "b_mm": 120.0, "tw_mm": 6.2, "tf_mm": 9.8,
        "area_cm2": 39.1, "I_y_cm4": 3892.0, "I_z_cm4": 284.0,
        "wel_y_cm3": 324.3, "wpl_y_cm3": 366.6,
    },
    "IPE270": {
        "h_mm": 270.0, "b_mm": 135.0, "tw_mm": 6.6, "tf_mm": 10.2,
        "area_cm2": 45.9, "I_y_cm4": 5790.0, "I_z_cm4": 420.0,
        "wel_y_cm3": 428.9, "wpl_y_cm3": 484.0,
    },
    "IPE300": {
        "h_mm": 300.0, "b_mm": 150.0, "tw_mm": 7.1, "tf_mm": 10.7,
        "area_cm2": 53.8, "I_y_cm4": 8356.0, "I_z_cm4": 604.0,
        "wel_y_cm3": 557.1, "wpl_y_cm3": 628.4,
    },
    "IPE330": {
        "h_mm": 330.0, "b_mm": 160.0, "tw_mm": 7.5, "tf_mm": 11.5,
        "area_cm2": 62.6, "I_y_cm4": 11770.0, "I_z_cm4": 788.0,
        "wel_y_cm3": 713.1, "wpl_y_cm3": 804.3,
    },
    "IPE360": {
        "h_mm": 360.0, "b_mm": 170.0, "tw_mm": 8.0, "tf_mm": 12.7,
        "area_cm2": 72.7, "I_y_cm4": 16270.0, "I_z_cm4": 1043.0,
        "wel_y_cm3": 903.6, "wpl_y_cm3": 1019.0,
    },
    "IPE400": {
        "h_mm": 400.0, "b_mm": 180.0, "tw_mm": 8.6, "tf_mm": 13.5,
        "area_cm2": 84.5, "I_y_cm4": 23130.0, "I_z_cm4": 1318.0,
        "wel_y_cm3": 1156.0, "wpl_y_cm3": 1307.0,
    },
    "IPE450": {
        "h_mm": 450.0, "b_mm": 190.0, "tw_mm": 9.4, "tf_mm": 14.6,
        "area_cm2": 98.8, "I_y_cm4": 33740.0, "I_z_cm4": 1676.0,
        "wel_y_cm3": 1500.0, "wpl_y_cm3": 1702.0,
    },
    "IPE500": {
        "h_mm": 500.0, "b_mm": 200.0, "tw_mm": 10.2, "tf_mm": 16.0,
        "area_cm2": 115.5, "I_y_cm4": 48200.0, "I_z_cm4": 2142.0,
        "wel_y_cm3": 1928.0, "wpl_y_cm3": 2194.0,
    },
    "IPE550": {
        "h_mm": 550.0, "b_mm": 210.0, "tw_mm": 11.1, "tf_mm": 17.2,
        "area_cm2": 134.4, "I_y_cm4": 67120.0, "I_z_cm4": 2668.0,
        "wel_y_cm3": 2441.0, "wpl_y_cm3": 2787.0,
    },
    "IPE600": {
        "h_mm": 600.0, "b_mm": 220.0, "tw_mm": 12.0, "tf_mm": 19.0,
        "area_cm2": 156.0, "I_y_cm4": 92080.0, "I_z_cm4": 3387.0,
        "wel_y_cm3": 3069.0, "wpl_y_cm3": 3512.0,
    },
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        clean = value.strip().replace(",", ".")
        try:
            return float(clean)
        except ValueError:
            return None
    return None


def _pick(row: dict[str, Any], aliases: list[str]) -> float | None:
    for alias in aliases:
        if alias in row:
            candidate = _as_float(row[alias])
            if candidate is not None:
                return candidate

    lowered = {str(k).lower(): v for k, v in row.items()}
    for alias in aliases:
        candidate = _as_float(lowered.get(alias.lower()))
        if candidate is not None:
            return candidate
    return None


def _normalize_eurocodepy_row(row: dict[str, Any]) -> dict[str, float] | None:
    """Normalize a eurocodepy profile row to our internal format.

    eurocodepy stores dimensions in cm; we convert to mm for h, b, tw, tf.
    Section moduli (Wel, Wpl) and area stay in cm³ / cm² / cm⁴.
    """
    # Dimensions are in cm in eurocodepy — convert to mm
    h_cm = _pick(row, ["h", "h_mm", "depth"])
    b_cm = _pick(row, ["b", "b_mm", "width"])
    tw_cm = _pick(row, ["tw", "tw_mm"])
    tf_cm = _pick(row, ["tf", "tf_mm"])
    area_cm2 = _pick(row, ["A", "A_cm2", "area", "area_cm2"])
    I_y_cm4 = _pick(row, ["Iy", "I_y_cm4"])
    I_z_cm4 = _pick(row, ["Iz", "I_z_cm4"])
    wel_y_cm3 = _pick(row, ["Wely", "Wel_y", "Wy", "W_el_y", "wel_y_cm3"])
    wpl_y_cm3 = _pick(row, ["Wply", "Wpl_y", "Wpy", "W_pl_y", "wpl_y_cm3"])

    required = [h_cm, b_cm, tw_cm, tf_cm, area_cm2, wel_y_cm3, wpl_y_cm3]
    if any(value is None for value in required):
        return None

    # Detect unit: if h < 10 for a typical IPE section, it's probably cm
    # (IPE80 has h=8cm, the smallest; IPE600 has h=60cm)
    cm_to_mm = 10.0 if float(h_cm) < 100.0 else 1.0

    result: dict[str, float] = {
        "h_mm": float(h_cm) * cm_to_mm,
        "b_mm": float(b_cm) * cm_to_mm,
        "tw_mm": float(tw_cm) * cm_to_mm,
        "tf_mm": float(tf_cm) * cm_to_mm,
        "area_cm2": float(area_cm2),
        "wel_y_cm3": float(wel_y_cm3),
        "wpl_y_cm3": float(wpl_y_cm3),
    }
    if I_y_cm4 is not None:
        result["I_y_cm4"] = float(I_y_cm4)
    if I_z_cm4 is not None:
        result["I_z_cm4"] = float(I_z_cm4)
    return result


def load_ipe_sections() -> tuple[dict[str, dict[str, float]], str]:
    """Load IPE section database, preferring eurocodepy if available."""
    # Primary source: eurocodepy profile database.
    try:
        import eurocodepy as ec  # type: ignore

        db = getattr(ec, "db", None)
        if isinstance(db, dict):
            steel = db.get("SteelProfiles")
            if isinstance(steel, dict):
                euro_i = steel.get("EuroI")
                # eurocodepy stores EuroI as a LIST of dicts, not a dict
                if isinstance(euro_i, list):
                    sections: dict[str, dict[str, float]] = {}
                    for raw_row in euro_i:
                        if not isinstance(raw_row, dict):
                            continue
                        name = str(raw_row.get("Section", "")).upper().replace(" ", "")
                        if not name.startswith("IPE"):
                            continue
                        normalized = _normalize_eurocodepy_row(raw_row)
                        if normalized:
                            sections[name] = normalized
                    if sections:
                        return sections, "eurocodepy.db.SteelProfiles.EuroI"
    except Exception:  # noqa: BLE001
        pass

    return FALLBACK_IPE_SECTIONS.copy(), "fallback_embedded"
