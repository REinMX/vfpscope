"""Derived quantities: delta-P curves, gradient, ΔP decomposition (spec 8.6)."""

from __future__ import annotations

import numpy as np

from .model import VfpTable


def dp_curve(table: VfpTable, thp_index: int = 0, wfr: int = 0, gfr: int = 0,
             alq: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(FLO values, BHP - THP) for one slice — the pressure drop across the
    tubing (well role) or across the branch (branch role)."""
    flo = table.axes["FLO"].values
    thp = table.axes["THP"].values[thp_index]
    dp = table.data[thp_index, wfr, gfr, alq, :] - thp
    return flo, dp


def gradient(table: VfpTable, thp_index: int = 0, wfr: int = 0, gfr: int = 0,
             alq: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """d(BHP)/dFLO at midpoints for one slice."""
    flo, dp = dp_curve(table, thp_index, wfr, gfr, alq)
    mid = 0.5 * (flo[:-1] + flo[1:])
    grad = np.diff(dp) / np.diff(flo)
    return mid, grad


def fit_pressure_drop(table: VfpTable, thp_index: int = 0, wfr: int = 0,
                      gfr: int = 0, alq: int = 0) -> dict | None:
    """Fit dP = a + b*Q^n to a curve; returns {a, b, n} or None.

    The intercept approximates the hydrostatic contribution and the exponent
    the friction regime — n far from ~2 on a single-phase gas branch signals
    a bad table (spec 8.6).
    """
    flo, dp = dp_curve(table, thp_index, wfr, gfr, alq)
    if flo.size < 4 or np.any(dp <= 0):
        return None
    try:
        from scipy.optimize import curve_fit

        def model(q, a, b, n):
            return a + b * np.power(q, n)

        p0 = [float(dp[0]), max(float((dp[-1] - dp[0]) / flo[-1] ** 2), 1e-12), 2.0]
        (a, b, n), _ = curve_fit(
            model, flo, dp, p0=p0,
            bounds=([0.0, 0.0, 0.5], [np.inf, np.inf, 3.5]),
            maxfev=20000,
        )
        return {"a": float(a), "b": float(b), "n": float(n)}
    except Exception:
        return None


def turning_points(table: VfpTable) -> list[dict]:
    """Turning points (unstable limb) per curve: {thp,wfr,gfr,alq,flo,value}."""
    flo = table.axes["FLO"].values
    out: list[dict] = []
    for ti in range(table.axes["THP"].values.size):
        for w in range(table.axes["WFR"].values.size):
            for g in range(table.axes["GFR"].values.size):
                for a in range(table.axes["ALQ"].values.size):
                    curve = table.data[ti, w, g, a, :]
                    d = np.sign(np.diff(curve))
                    turns = np.where(np.diff(d) != 0)[0]
                    for k in turns:
                        k = int(k) + 1
                        out.append({
                            "thp": ti, "wfr": w, "gfr": g, "alq": a,
                            "flo": float(flo[k]), "value": float(curve[k]),
                        })
    return out
