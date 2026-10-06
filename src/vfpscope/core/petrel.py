"""Petrel / Eclipse simulation view of an already-parsed VFP deck.

Does not generate tables and does not claim an Eclipse executable result.
Eclipse (the Petrel simulation run) linearly extrapolates outside a multi-point
axis. A single THP is different: the manual sets dBHP/dTHP = 1, so bottom-hole
pressure moves one-for-one with tubing-head pressure. That slope is not applied
to a tubing-head temperature table. Length-1 WFR, GFR and ALQ have no slope.
``linear_bhp`` is that rule applied to the table. It is not an Eclipse run.
``edge_bhp`` is the clamped table boundary from ``lookup``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np

from .interp import lookup
from .model import AXIS_ORDER, VfpTable
from .qc.engine import run_qc

_FLO = {
    "OIL": "Oil rate",
    "LIQ": "Liquid rate",
    "GAS": "Gas rate",
    "WG": "Wet-gas rate",
    "TM": "Total molar rate",
    "WAT": "Water rate",
}
_WFR = {
    "WOR": "Water-oil ratio",
    "WCT": "Water cut",
    "WGR": "Water-gas ratio",
    "WWR": "Water-wet-gas ratio",
    "WTF": "Water mole fraction",
}
_GFR = {
    "GOR": "Gas-oil ratio",
    "GLR": "Gas-liquid ratio",
    "OGR": "Oil-gas ratio",
    "MMW": "Mean molecular weight",
}
_ALQ = {
    "GRAT": "Gas-lift rate",
    "IGLR": "Injection gas-liquid ratio",
    "TGLR": "Total gas-liquid ratio",
    "PUMP": "Pump rating",
    "COMP": "Compressor power",
    "DENO": "Oil surface density",
    "DENG": "Gas surface density",
    "BEAN": "Choke size",
    "": "Artificial-lift quantity",
}

_IMPLICATIONS = {
    "UNSTABLE_BRANCH": (
        "A turning point in bottom-hole pressure versus rate can give two solutions "
        "at one tubing-head pressure. Petrel/Eclipse THP control may oscillate or "
        "fail to converge. Regenerate the table or stay off that rate."
    ),
    "THP_MONOTONIC": (
        "Bottom-hole pressure falls when tubing-head pressure rises. Eclipse treats "
        "that as a non-physical VFP response. Do not use this table in the Petrel case."
    ),
    "CROSSING": (
        "Tubing-head-pressure curves cross. Eclipse can pick the wrong branch and "
        "the well may not converge under THP control."
    ),
    "MISSING_TABLE": (
        "A well or branch points at a VFP table number that is not in the deck. "
        "The Petrel run stops when that control is read."
    ),
    "UNITS_MISMATCH": (
        "The table unit system does not match the simulation case. Eclipse rejects "
        "the table. Re-export it in the Petrel project units."
    ),
    "DATUM_MISMATCH": (
        "Table datum differs from the well reference depth. Eclipse applies a "
        "hydrostatic correction; a large gap usually means the wrong table was assigned."
    ),
    "BHP_LT_THP": (
        "Tabulated pressure is below tubing-head pressure, so the tubing pressure "
        "drop is negative. Check the export before using THP control."
    ),
    "DEAD_TABLE": (
        "No well or flowline uses this table. It does not affect the run unless a "
        "later development-strategy step points at it."
    ),
    "ALQ_GASLIFT": (
        "The well has gas lift, but the VFP table has only one lift value. Changing "
        "the lift rate in Petrel will not change bottom-hole pressure."
    ),
    "ROLE_CONFLICT": (
        "The same table number is both a well lift table and a network branch. "
        "That is almost always an export mix-up."
    ),
    "CLAMP_FRACTION": (
        "The simulated well spends time outside the table. Eclipse extrapolates "
        "there; extend the table before trusting the Petrel run."
    ),
}


def axis_caption(table: VfpTable, axis: str) -> str:
    """Petrel-facing name for one axis, with its unit when known."""
    kind = table.axes[axis].kind
    unit = table.axes[axis].unit
    if axis == "THP":
        name = "Outlet pressure" if table.role == "BRANCH" else "Tubing-head pressure"
    elif axis == "FLO":
        name = _FLO.get(kind, f"Flow rate ({kind})")
    elif axis == "WFR":
        name = _WFR.get(kind, f"Water fraction ({kind})")
    elif axis == "GFR":
        name = _GFR.get(kind, f"Gas fraction ({kind})")
    elif axis == "ALQ":
        name = _ALQ.get(kind, "Artificial-lift quantity")
    else:
        name = axis
    return f"{name} ({unit})" if unit else name


_THT_UNITS = {"METRIC": "°C", "FIELD": "°F", "LAB": "°C", "PVT-M": "°C"}


def quantity_unit(table: VfpTable) -> str:
    """Unit of the looked-up value. Temperature tables are not in pressure units."""
    if table.tabulated == "TEMP":
        return _THT_UNITS.get(table.unit_system, "")
    return table.axes["THP"].unit


def quantity_caption(table: VfpTable) -> str:
    """Name of the value the simulator looks up."""
    if table.tabulated == "TEMP":
        return "Tubing-head temperature"
    if table.role == "BRANCH":
        return "Inlet pressure"
    return "Bottom-hole pressure"


AxisStatus = Literal["inside", "low", "high", "independent", "unit_slope"]


@dataclass(frozen=True)
class AxisQuery:
    name: str
    label: str
    kind: str
    unit: str
    value: float
    low: float
    high: float
    status: AxisStatus


@dataclass(frozen=True)
class OperatingPoint:
    linear_bhp: float
    edge_bhp: float
    inside: bool
    note: str
    axes: tuple[AxisQuery, ...]
    quantity: str
    unit: str

    def axis(self, name: str) -> AxisQuery:
        for item in self.axes:
            if item.name == name:
                return item
        raise KeyError(name)


def _bracket(values: np.ndarray, x: float) -> tuple[int, int, float, AxisStatus]:
    if values.size == 1:
        return 0, 0, 0.0, "independent"
    if x < values[0]:
        span = float(values[1] - values[0])
        return 0, 1, (x - float(values[0])) / span, "low"
    if x > values[-1]:
        i0 = values.size - 2
        span = float(values[-1] - values[i0])
        return i0, values.size - 1, (x - float(values[i0])) / span, "high"
    i0 = int(np.searchsorted(values, x, side="right") - 1)
    i0 = min(max(i0, 0), values.size - 2)
    i1 = i0 + 1
    span = float(values[i1] - values[i0])
    return i0, i1, (x - float(values[i0])) / span, "inside"


def _axis_status(table: VfpTable, name: str, values: np.ndarray, x: float) -> AxisStatus:
    if values.size == 1 and name == "THP" and table.tabulated != "TEMP":
        if np.isclose(x, float(values[0])):
            return "inside"
        return "unit_slope"
    return _bracket(values, x)[3]


def _linear_value(table: VfpTable, coords: dict[str, float]) -> float:
    brackets = {name: _bracket(table.axes[name].values, coords[name]) for name in AXIS_ORDER}
    value = 0.0
    for corner in product((0, 1), repeat=len(AXIS_ORDER)):
        weight = 1.0
        index: list[int] = []
        for name, bit in zip(AXIS_ORDER, corner, strict=True):
            i0, i1, t, _status = brackets[name]
            index.append(i1 if bit else i0)
            weight *= t if bit else (1.0 - t)
        value += weight * float(table.data[tuple(index)])
    thp = table.axes["THP"].values
    if thp.size == 1 and table.tabulated != "TEMP":
        # VFPPROD/VFPINJ record 3: one THP => dBHP/dTHP = 1. Not used for THT.
        value += coords["THP"] - float(thp[0])
    return value


def _note(table: VfpTable, axes: tuple[AxisQuery, ...]) -> str:
    quantity = quantity_caption(table).lower()
    outside = [item for item in axes if item.status in {"low", "high"}]
    slope = [item for item in axes if item.status == "unit_slope"]
    if not outside and not slope:
        return (
            f"Inside the table. {quantity_caption(table)} is multilinear interpolation. "
            "Eclipse uses the same interpolation inside the table. This is not an Eclipse run."
        )
    parts = []
    if outside:
        names = ", ".join(f"{item.label} ({item.status})" for item in outside)
        parts.append(
            f"Outside the table on {names}. Eclipse linearly extrapolates and can return an "
            f"unrealistic {quantity}. linear_bhp continues the edge segment. edge_bhp is the "
            "table boundary, not the Eclipse answer. This is not an Eclipse run."
        )
    if slope:
        parts.append(
            "This table has one tubing-head pressure. Eclipse treats the derivative of "
            "bottom-hole pressure with respect to tubing-head pressure as 1. "
            "This is not an Eclipse run."
        )
    return " ".join(parts)


def evaluate_operating_point(
    table: VfpTable,
    *,
    flo: float,
    thp: float,
    wfr: float | None = None,
    gfr: float | None = None,
    alq: float | None = None,
) -> OperatingPoint:
    """Look up one development-strategy operating point.

    Missing ratio or lift values use the first tabulated entry. Length-1 WFR,
    GFR and ALQ stay independent. A single THP on a pressure table uses unit slope.
    """
    given = {"FLO": flo, "THP": thp, "WFR": wfr, "GFR": gfr, "ALQ": alq}
    coords: dict[str, float] = {}
    axes: list[AxisQuery] = []
    for name in AXIS_ORDER:
        values = table.axes[name].values
        raw = given[name]
        value = float(values[0] if raw is None else raw)
        coords[name] = value
        status = _axis_status(table, name, values, value)
        axes.append(
            AxisQuery(
                name=name,
                label=axis_caption(table, name),
                kind=table.axes[name].kind,
                unit=table.axes[name].unit,
                value=value,
                low=float(values[0]),
                high=float(values[-1]),
                status=status,
            )
        )
    axis_rows = tuple(axes)
    inside = all(item.status in {"inside", "independent"} for item in axis_rows)
    edge = lookup(
        table,
        flo=coords["FLO"],
        thp=coords["THP"],
        wfr=coords["WFR"],
        gfr=coords["GFR"],
        alq=coords["ALQ"],
    )
    return OperatingPoint(
        linear_bhp=_linear_value(table, coords),
        edge_bhp=float(edge.value),
        inside=inside,
        note=_note(table, axis_rows),
        axes=axis_rows,
        quantity=quantity_caption(table),
        unit=quantity_unit(table),
    )


@dataclass(frozen=True)
class WellAssignment:
    well: str
    phase: str
    datum: float | None
    table_number: int
    table_present: bool
    table_kind: str | None
    table_datum: float | None
    gas_lift: bool


@dataclass(frozen=True)
class SimulationBrief:
    gate: Literal["RUN", "REVIEW", "DO_NOT_RUN"]
    headline: str
    unit_system: str
    n_tables: int
    n_errors: int
    n_warnings: int
    wells: tuple[WellAssignment, ...]
    unassigned_tables: tuple[int, ...]


def simulation_brief(deck) -> SimulationBrief:
    """Run-gate for a Petrel Eclipse export. Calls the QC engine; does not mutate the deck."""
    findings = run_qc(deck)
    n_errors = sum(1 for item in findings if item.severity == "ERROR")
    n_warnings = sum(1 for item in findings if item.severity == "WARNING")
    if n_errors:
        gate: Literal["RUN", "REVIEW", "DO_NOT_RUN"] = "DO_NOT_RUN"
        headline = (
            f"Do not run — {n_errors} error(s) in the VFP tables. "
            "Petrel/Eclipse will reject the deck or mis-apply the table."
        )
    elif n_warnings:
        gate = "REVIEW"
        headline = (
            f"Review before the Petrel run — {n_warnings} warning(s). "
            "The deck can load, but THP control may be unstable or incomplete."
        )
    else:
        gate = "RUN"
        headline = (
            "No blocking VFP findings. Still confirm the table covers the rates, "
            "THP, water cut, GOR and lift the development strategy will use."
        )
    wells: list[WellAssignment] = []
    for well, number in sorted(deck.well_vfp.items()):
        table = deck.tables.get(number)
        wells.append(
            WellAssignment(
                well=well,
                phase=deck.well_phases.get(well, ""),
                datum=deck.wells_datum.get(well),
                table_number=number,
                table_present=table is not None,
                table_kind=None if table is None else table.kind,
                table_datum=None if table is None else table.datum_depth,
                gas_lift=well in deck.gaslift_wells,
            )
        )
    unassigned = tuple(number for number in deck.table_order if not deck.tables[number].consumers)
    return SimulationBrief(
        gate=gate,
        headline=headline,
        unit_system=deck.unit_system,
        n_tables=len(deck.tables),
        n_errors=n_errors,
        n_warnings=n_warnings,
        wells=tuple(wells),
        unassigned_tables=unassigned,
    )


def assignments_csv(brief: SimulationBrief) -> str:
    """CSV of well → VFP table assignments. Datum is in the deck unit system."""
    lines = ["well,phase,datum,vfp_table,table_present,table_kind,table_datum,gas_lift"]
    for well in brief.wells:
        datum = "" if well.datum is None else f"{well.datum:g}"
        table_datum = "" if well.table_datum is None else f"{well.table_datum:g}"
        kind = well.table_kind or ""
        lines.append(
            f"{well.well},{well.phase},{datum},{well.table_number},"
            f"{str(well.table_present).lower()},{kind},{table_datum},{str(well.gas_lift).lower()}"
        )
    return "\n".join(lines) + "\n"


def finding_implication(check_id: str) -> str:
    """What the finding means for a Petrel/Eclipse run. Does not propose a silent repair."""
    return _IMPLICATIONS.get(
        check_id,
        "Check this finding before the Petrel run. VFPScope does not repair the table.",
    )
