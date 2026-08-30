"""Coverage analysis (M5): simulated operating envelope vs table ranges.

Loads a run's summary (via ``resdata``, optional) and, per consuming well,
extracts the vectors matching the table's FLO/WFR/GFR/ALQ/THP types, runs the
same clamped lookup the simulator uses, and reports per-axis clamping
fractions and range exceedances.

``resdata`` is an optional dependency: the analysis core works against any
object exposing ``numpy_vector(key)`` (see SummaryFake in the tests), so the
module stays importable on machines without resdata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .interp import _lookup_many
from .model import AXIS_ORDER, VfpTable

# summary vector names per axis kind (first available wins; None = not mappable)
_VECTORS: dict[str, dict[str, str | tuple[str, ...] | None]] = {
    "FLO": {
        "OIL": "WOPR",
        "LIQ": ("WOPR", "WWPR"),
        "GAS": "WGPR",
        "WG": ("WOPR", "WWPR", "WGPR"),
        "TM": ("WOPR", "WWPR", "WGPR"),
        "WAT": "WWIR",
    },
    "THP": {"THP": "WTHP"},
    "WFR": {
        "WOR": ("WWPR", "WOPR"),
        "WCT": "WWCT",
        "WGR": "WWGR",
        "WWR": None,
        "WTF": None,
    },
    "GFR": {
        "GOR": "WGOR",
        "GLR": "WGLR",
        "OGR": None,
        "MMW": None,
    },
    "ALQ": {
        "GRAT": "WGLIR",
        "IGLR": None,
        "TGLR": None,
        "PUMP": None,
        "COMP": None,
        "DENO": None,
        "DENG": None,
        "BEAN": None,
        "": None,
    },
}


class CoverageUnavailable(RuntimeError):
    """resdata (or the summary file) is not available."""


@dataclass
class CoverageReport:
    """Per-well coverage of a table by one run's operating envelope."""

    table_number: int
    well: str
    n_timesteps: int
    axis_values: dict[str, np.ndarray | None] = field(default_factory=dict)
    clamped_low: dict[str, np.ndarray | None] = field(default_factory=dict)
    clamped_high: dict[str, np.ndarray | None] = field(default_factory=dict)
    bhp: np.ndarray | None = None  # WBHP (for overlay scatter)

    @property
    def axes_with_data(self) -> list[str]:
        return [a for a in AXIS_ORDER if self.axis_values.get(a) is not None]

    def fraction_clamped(self, axis: str) -> float:
        lo = self.clamped_low.get(axis)
        hi = self.clamped_high.get(axis)
        if lo is None or hi is None:
            return 0.0
        return float(np.mean(lo | hi))

    def fraction_clamped_low(self, axis: str) -> float:
        lo = self.clamped_low.get(axis)
        return float(np.mean(lo)) if lo is not None else 0.0

    def fraction_clamped_high(self, axis: str) -> float:
        hi = self.clamped_high.get(axis)
        return float(np.mean(hi)) if hi is not None else 0.0

    def run_max_over_table(self, axis: str, table: VfpTable) -> float | None:
        v = self.axis_values.get(axis)
        tmax = float(table.axes[axis].values[-1])
        if v is None or tmax <= 0:
            return None
        return float(v.max()) / tmax

    def run_min_under_table(self, axis: str, table: VfpTable) -> float | None:
        v = self.axis_values.get(axis)
        tmin = float(table.axes[axis].values[0])
        if v is None or tmin <= 0:
            return None
        return float(v.min()) / tmin


def _resolve(smry, well: str, spec) -> np.ndarray | None:
    if spec is None:
        return None
    names = spec if isinstance(spec, tuple) else (spec,)
    try:
        parts = []
        for n in names:
            key = f"{n}:{well}"
            try:
                vec = np.asarray(smry.numpy_vector(key), dtype=float)
            except (KeyError, TypeError, ValueError):
                return None
            parts.append(vec)
        if not parts:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out
    except Exception:
        return None


def _ratio_from_components(parts: list[np.ndarray]) -> np.ndarray:
    """WOR-style ratio: num/den with den==0 guarded to 0."""
    num, den = parts[0], parts[1]
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out


def coverage_from_summary(table: VfpTable, well: str, smry) -> CoverageReport:
    """Evaluate one well's run envelope against one table."""
    n = None
    axis_values: dict[str, np.ndarray | None] = {}
    for a in AXIS_ORDER:
        kind = table.axes[a].kind
        spec = _VECTORS[a].get(kind)
        if a == "WFR" and kind == "WOR":
            num = _resolve(smry, well, "WWPR")
            den = _resolve(smry, well, "WOPR")
            vec = _ratio_from_components([num, den]) if num is not None and den is not None else None
        else:
            vec = _resolve(smry, well, spec)
        axis_values[a] = vec
        if vec is not None:
            n = vec.size
    if n is None:
        raise CoverageUnavailable(
            f"no summary vectors match table {table.number} for well {well}"
        )
    wbhp = _resolve(smry, well, "WBHP")

    coords = {}
    for a in AXIS_ORDER:
        v = axis_values[a]
        coords[a] = v if v is not None else np.full(n, float(table.axes[a].values[0]))
    _, masks = _lookup_many(table, coords)

    # masks are always computed; keep only axes with data
    clamped_low = {a: masks[a][0] if axis_values[a] is not None else None for a in AXIS_ORDER}
    clamped_high = {a: masks[a][1] if axis_values[a] is not None else None for a in AXIS_ORDER}
    return CoverageReport(
        table_number=table.number,
        well=well,
        n_timesteps=n,
        axis_values=axis_values,
        clamped_low=clamped_low,
        clamped_high=clamped_high,
        bhp=wbhp,
    )


def load_summary(path: str | Path):
    """Load a summary file via resdata (case stem without extension)."""
    p = Path(path)
    stem = str(p.with_suffix(""))
    try:
        from resdata.summary import Summary
    except ImportError as e:  # pragma: no cover
        raise CoverageUnavailable("resdata is not installed (pip install vfpscope[resdata])") from e
    return Summary(stem)


def coverage_for_deck(deck, smry_path: str | Path) -> dict[int, list[CoverageReport]]:
    """Coverage reports for every (table, consuming well) pair in a deck."""
    smry = load_summary(smry_path)
    out: dict[int, list[CoverageReport]] = {}
    for no in deck.table_order:
        t = deck.tables[no]
        reports = []
        for well in t.consumers.wells:
            try:
                reports.append(coverage_from_summary(t, well, smry))
            except CoverageUnavailable:
                continue
        if reports:
            out[no] = reports
    return out
