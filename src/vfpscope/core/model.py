"""Domain model for VFP tables.

CANONICAL AXIS ORDER
====================

``data`` is stored with shape ``(NTHP, NWFR, NGFR, NALQ, NFLO)`` — the fixed
canonical axis order ``(THP, WFR, GFR, ALQ, FLO)`` — and never deviates.
This matches the deck record ordering (each data record carries the
``(THP, WFR, GFR, ALQ)`` indices followed by ``NFLO`` tabulated values), so
the parser places values with zero transposition risk and every consumer
(interpolation, QC, figures) indexes the same way.

Index conversion: deck data-record indices are **1-based**; the parser
converts them to 0-based exactly once, at the parse boundary.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

AXIS_ORDER = ("THP", "WFR", "GFR", "ALQ", "FLO")

# --- valid enumerations (mirrors res2df._vfpdefs / OPM parser definitions) ---
VFPPROD_FLO_TYPES = ("OIL", "LIQ", "GAS", "WG", "TM")
VFPINJ_FLO_TYPES = ("OIL", "WAT", "GAS", "WG", "TM")
WFR_TYPES = ("WOR", "WCT", "WGR", "WWR", "WTF")
GFR_TYPES = ("GOR", "GLR", "OGR", "MMW")
THP_TYPES = ("THP",)
ALQ_TYPES = ("GRAT", "IGLR", "TGLR", "PUMP", "COMP", "DENO", "DENG", "BEAN", "")
UNIT_SYSTEMS = ("METRIC", "FIELD", "LAB", "PVT-M")
TABULATED_TYPES = ("BHP", "TEMP")

SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "ERROR": 2}

# --- axis units per unit system (mirrors res2df VFPPROD_UNITS / VFPINJ_UNITS) ---
_UNIT_TABLES: dict[str, dict[str, dict[str, str]]] = {
    "METRIC": {
        "FLO": {"OIL": "sm3/day", "LIQ": "sm3/day", "GAS": "sm3/day", "WG": "sm3/day", "TM": "kg-M/day", "WAT": "sm3/day"},
        "THP": {"THP": "barsa"},
        "WFR": {"WOR": "sm3/sm3", "WCT": "sm3/sm3", "WGR": "sm3/sm3", "WWR": "sm3/sm3", "WTF": ""},
        "GFR": {"GOR": "sm3/sm3", "GLR": "sm3/sm3", "OGR": "sm3/sm3", "MMW": "kg/kg-M"},
        "ALQ": {
            "GRAT": "sm3/day", "IGLR": "sm3/sm3", "TGLR": "sm3/sm3", "PUMP": "", "COMP": "",
            "DENO": "kg/m3", "DENG": "kg/m3", "BEAN": "mm", "": "",
        },
    },
    "FIELD": {
        "FLO": {"OIL": "stb/day", "LIQ": "stb/day", "GAS": "Mscf/day", "WG": "lb-M/day", "TM": "lb-M/day", "WAT": "stb/day"},
        "THP": {"THP": "psia"},
        "WFR": {"WOR": "stb/stb", "WCT": "stb/stb", "WGR": "stb/Mscf", "WWR": "stb/Mscf", "WTF": ""},
        "GFR": {"GOR": "Mscf/stb", "GLR": "Mscf/stb", "OGR": "stb/Mscf", "MMW": "lb/lb-M"},
        "ALQ": {
            "GRAT": "Mscf/day", "IGLR": "Mscf/stb", "TGLR": "Mscf/stb", "PUMP": "", "COMP": "",
            "DENO": "lb/ft3", "DENG": "lb/ft3", "BEAN": "1/64", "": "",
        },
    },
    "LAB": {
        "FLO": {"OIL": "scc/hr", "LIQ": "scc/hr", "GAS": "scc/hr", "WG": "scc/hr", "TM": "lb-M/day", "WAT": "scc/hr"},
        "THP": {"THP": "atma"},
        "WFR": {"WOR": "scc/scc", "WCT": "scc/scc", "WGR": "scc/scc", "WWR": "scc/scc", "WTF": ""},
        "GFR": {"GOR": "scc/scc", "GLR": "scc/scc", "OGR": "scc/scc", "MMW": "lb/lb-M"},
        "ALQ": {
            "GRAT": "scc/hr", "IGLR": "scc/scc", "TGLR": "scc/scc", "PUMP": "", "COMP": "",
            "DENO": "gm/cc", "DENG": "gm/cc", "BEAN": "mm", "": "",
        },
    },
    "PVT-M": {
        "FLO": {"OIL": "sm3/day", "LIQ": "sm3/day", "GAS": "sm3/day", "WG": "sm3/day", "TM": "kg-M/day", "WAT": "sm3/day"},
        "THP": {"THP": "atma"},
        "WFR": {"WOR": "sm3/sm3", "WCT": "sm3/sm3", "WGR": "sm3/sm3", "WWR": "sm3/sm3", "WTF": ""},
        "GFR": {"GOR": "sm3/sm3", "GLR": "sm3/sm3", "OGR": "sm3/sm3", "MMW": "kg/kg-M"},
        "ALQ": {
            "GRAT": "sm3/day", "IGLR": "sm3/sm3", "TGLR": "sm3/sm3", "PUMP": "", "COMP": "",
            "DENO": "kg/m3", "DENG": "kg/m3", "BEAN": "mm", "": "",
        },
    },
}

# map axis-name -> unit-table key
_AXIS_UNIT_KEY = {"FLO": "FLO", "THP": "THP", "WFR": "WFR", "GFR": "GFR", "ALQ": "ALQ"}


def resolve_axis_unit(unit_system: str, axis: str, kind: str) -> str:
    """Resolve the display unit for an axis from its unit system and kind."""
    if unit_system not in _UNIT_TABLES:
        return ""
    return _UNIT_TABLES[unit_system].get(_AXIS_UNIT_KEY[axis], {}).get(kind, "")


class SourceRef(BaseModel):
    """File + line provenance for parser errors and display."""

    model_config = ConfigDict(frozen=True)

    path: str
    line: int

    def __str__(self) -> str:
        return f"{self.path}:{self.line}"


class Consumers(BaseModel):
    """Which wells/branches reference a table (role inference result)."""

    model_config = ConfigDict(frozen=True)

    wells: tuple[str, ...] = ()
    branches: tuple[tuple[str, str], ...] = ()  # (downtree_node, uptree_node)

    def __bool__(self) -> bool:
        return bool(self.wells or self.branches)


class VfpAxis(BaseModel):
    """One interpolation axis of a VFP table.

    ``values`` must be 1-D and strictly increasing — validated at the parse
    boundary; a table with a broken axis is reported as a parse finding and
    never built.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    kind: str
    values: np.ndarray
    unit: str = ""

    @model_validator(mode="after")
    def _validate(self) -> VfpAxis:
        v = self.values
        if not isinstance(v, np.ndarray) or v.ndim != 1 or v.size == 0:
            raise ValueError(f"axis {self.name}: values must be a non-empty 1-D array")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"axis {self.name}: values must be finite")
        if not np.all(np.diff(v) > 0):
            raise ValueError(f"axis {self.name}: values must be strictly increasing")
        return self


class Finding(BaseModel):
    """A structured QC finding. Formatting is the caller's job."""

    model_config = ConfigDict(frozen=True)

    check_id: str
    severity: Literal["ERROR", "WARNING", "INFO"]
    table_number: int
    message: str
    locus: dict = Field(default_factory=dict)  # axis indices / slice / file+line
    plot_hint: dict | None = None  # enough for the UI to jump to the problem


class VfpTable(BaseModel):
    """One parsed VFP table (VFPPROD or VFPINJ).

    ``data`` shape: ``(NTHP, NWFR, NGFR, NALQ, NFLO)`` — canonical order
    ``(THP, WFR, GFR, ALQ, FLO)``. Never deviate (see module docstring).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    number: int
    kind: Literal["VFPPROD", "VFPINJ"]
    role: Literal["WELL", "BRANCH", "UNKNOWN"] = "UNKNOWN"
    datum_depth: float
    unit_system: Literal["METRIC", "FIELD", "LAB", "PVT-M", "DEFAULT"] = "DEFAULT"
    tabulated: Literal["BHP", "TEMP"] = "BHP"
    axes: dict[str, VfpAxis]
    data: np.ndarray
    source: SourceRef
    consumers: Consumers = Field(default_factory=Consumers)

    @model_validator(mode="after")
    def _validate(self) -> VfpTable:
        if set(self.axes) != set(AXIS_ORDER):
            raise ValueError(f"table {self.number}: axes must be exactly {AXIS_ORDER}")
        shape = tuple(len(self.axes[a].values) for a in AXIS_ORDER)
        if self.data.shape != shape:
            raise ValueError(
                f"table {self.number}: data shape {self.data.shape} != axes shape {shape}"
            )
        if self.kind == "VFPINJ" and any(
            self.axes[a].values.size != 1 for a in ("WFR", "GFR", "ALQ")
        ):
            # VFPINJ is a degenerate VFPPROD: extra axes must be length 1
            raise ValueError(
                f"table {self.number}: VFPINJ WFR/GFR/ALQ axes must have length 1"
            )
        return self

    # --- convenience accessors -------------------------------------------------
    @property
    def flo(self) -> VfpAxis:
        return self.axes["FLO"]

    @property
    def thp(self) -> VfpAxis:
        return self.axes["THP"]

    @property
    def wfr(self) -> VfpAxis:
        return self.axes["WFR"]

    @property
    def gfr(self) -> VfpAxis:
        return self.axes["GFR"]

    @property
    def alq(self) -> VfpAxis:
        return self.axes["ALQ"]

    @property
    def axis_lengths(self) -> dict[str, int]:
        return {a: self.axes[a].values.size for a in AXIS_ORDER}

    @property
    def n_data_records(self) -> int:
        """NTHP x NWFR x NGFR x NALQ — the expected number of data records."""
        n = 1
        for a in ("THP", "WFR", "GFR", "ALQ"):
            n *= self.axes[a].values.size
        return n

    @property
    def label(self) -> str:
        role = {"WELL": "well", "BRANCH": "branch", "UNKNOWN": "unassigned"}[self.role]
        return f"{self.kind} #{self.number} ({role})"

    def axis(self, name: str) -> VfpAxis:
        return self.axes[name]
