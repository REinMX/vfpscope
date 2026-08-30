"""Multilinear interpolation with simulator-matching clamp semantics.

``lookup`` mirrors what the simulator does with a VFP table:

* multilinear interpolation inside the hypercube;
* **clamping, not extrapolation**, at the edges — a query outside the table
  returns the edge value and reports ``clamped[axis]`` so callers can see
  that the answer came from a table boundary (the classic silent
  under-prediction of friction).

Degenerate axes (VFPINJ: WFR/GFR/ALQ length 1) interpolate to their single
value and are reported as clamped whenever the query is off that value.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .model import AXIS_ORDER, VfpTable


@dataclass(frozen=True)
class LookupResult:
    value: float | np.ndarray
    # scalar queries: per-axis "low" | "high" | None
    clamped: dict[str, str | None] = None  # type: ignore[assignment]
    # array queries: per-axis (low_mask, high_mask) bool arrays
    clamped_masks: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def any_clamped(self) -> bool:
        if self.clamped is not None:
            return any(v is not None for v in self.clamped.values())
        if self.clamped_masks:
            return any(
                bool(np.any(lo) or np.any(hi))
                for lo, hi in self.clamped_masks.values()
            )
        return False


def _axis_interp(values: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Bracketing indices, interpolation weights, clamp flags for one axis."""
    v = values
    low = x < v[0]
    high = x > v[-1]
    xc = np.clip(x, v[0], v[-1])
    if v.size == 1:
        z = np.zeros(x.shape, dtype=int)
        return z, z, np.zeros(x.shape), (low, high)
    i0 = np.searchsorted(v, xc, side="right") - 1
    i0 = np.clip(i0, 0, v.size - 2)
    i1 = i0 + 1
    denom = v[i1] - v[i0]
    t = np.zeros_like(xc)
    np.divide(xc - v[i0], denom, out=t, where=denom != 0)
    return i0, i1, t, (low, high)


def _lookup_many(table: VfpTable, coords: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    n = len(next(iter(coords.values())))
    i0s: dict[str, np.ndarray] = {}
    i1s: dict[str, np.ndarray] = {}
    ts: dict[str, np.ndarray] = {}
    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for a in AXIS_ORDER:
        x = coords[a]
        i0, i1, t, m = _axis_interp(table.axes[a].values, x)
        i0s[a] = i0
        i1s[a] = i1
        ts[a] = t
        masks[a] = m
    value = np.zeros(n)
    for corner in product((0, 1), repeat=len(AXIS_ORDER)):
        w = np.ones(n)
        idx: list[np.ndarray] = []
        for a, c in zip(AXIS_ORDER, corner):
            idx.append(i1s[a] if c else i0s[a])
            w = w * (ts[a] if c else (1.0 - ts[a]))
        value += w * table.data[tuple(idx)]
    return value, masks


def lookup(
    table: VfpTable,
    *,
    flo: float | np.ndarray,
    thp: float | np.ndarray,
    wfr: float | np.ndarray | None = None,
    gfr: float | np.ndarray | None = None,
    alq: float | np.ndarray | None = None,
) -> LookupResult:
    """Evaluate the table at a query point (scalar or equal-length arrays).

    Missing ratio/ALQ coordinates default to the **first** value of that axis
    (e.g. WCT=0, lowest GOR), which is the natural base case.
    """
    defaults = {a: table.axes[a].values[0] for a in ("WFR", "GFR", "ALQ")}
    given = {"WFR": wfr, "GFR": gfr, "ALQ": alq}
    coords: dict[str, np.ndarray] = {
        "FLO": np.asarray(flo, dtype=float),
        "THP": np.asarray(thp, dtype=float),
    }
    for a in ("WFR", "GFR", "ALQ"):
        v = given[a]
        coords[a] = np.asarray(defaults[a] if v is None else v, dtype=float)

    scalar = all(c.ndim == 0 for c in coords.values())
    if scalar:
        coords = {a: c.reshape(1) for a, c in coords.items()}
    else:
        n = coords["FLO"].size
        for a, c in coords.items():
            if c.ndim > 0 and c.size != n:
                raise ValueError(
                    "all query coordinates must be scalars or equal-length arrays"
                )
        coords = {
            a: (c if c.ndim > 0 else np.full(n, float(c))) for a, c in coords.items()
        }

    values, masks = _lookup_many(table, coords)
    if scalar:
        clamped = {a: ("low" if m[0][0] else "high" if m[1][0] else None) for a, m in masks.items()}
        return LookupResult(value=float(values[0]), clamped=clamped)
    return LookupResult(value=values, clamped_masks=masks)
