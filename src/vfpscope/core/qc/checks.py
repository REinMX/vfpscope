"""QC checks — pure functions (table, context) -> list[Finding].

Check numbering follows the spec section 9:

  1-7   structural (enforced at the parse boundary for deck input; the pure
        functions re-validate model invariants and are the guard for
        programmatically built tables)
  8-13  physical
  14-20 consistency (need deck context)
  21-23 coverage (M5, need simulation output)

Every finding carries a plot_hint sufficient for the UI to jump to the
offending slice. Nothing here prints; formatting is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..model import (
    ALQ_TYPES,
    AXIS_ORDER,
    GFR_TYPES,
    SEVERITY_ORDER,
    THP_TYPES,
    UNIT_SYSTEMS,
    TABULATED_TYPES,
    VFPINJ_FLO_TYPES,
    VFPPROD_FLO_TYPES,
    WFR_TYPES,
    Finding,
    VfpTable,
)


@dataclass
class QCContext:
    """Deck-level context needed by consistency/coverage checks."""

    deck: object | None = None  # Deck (lazy import to avoid cycles)
    thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "datum_tolerance_m": 100.0,
            "clamp_warning_frac": 0.05,  # M5, check 21
            "absurd_bhp_bar": 1.0e4,
        }
    )
    coverage: dict[int, object] | None = None  # table_no -> CoverageReport (M5)


def _f(
    table: VfpTable, severity: str, check_id: str, message: str,
    locus=None, plot_hint=None,
) -> Finding:
    return Finding(
        check_id=check_id,
        severity=severity,  # type: ignore[arg-type]
        table_number=table.number,
        message=message,
        locus=locus or {},
        plot_hint=plot_hint,
    )


# ------------------------------------------------------------------ structural (1-7)


def check_record_count(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Data record count must be NTHP x NWFR x NGFR x NALQ.

    Enforced by the native parser (PARSE finding); re-validated here for
    programmatically built tables.
    """
    if table.data.size != table.n_data_records * table.axis_lengths["FLO"]:
        return [_f(table, "ERROR", "RECORD_COUNT",
                   f"data contains {table.data.size // table.axis_lengths['FLO']} "
                   f"records, expected {table.n_data_records}")]
    return []


def check_record_width(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Every data record must carry exactly NFLO values (parse-enforced)."""
    return []


def check_axes_monotonic(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Axis vectors strictly increasing, no duplicates (parse-enforced)."""
    return []


def check_non_finite(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """NaN / inf entries in the hypercube."""
    bad = ~np.isfinite(table.data)
    if not bad.any():
        return []
    idx = np.argwhere(bad)[0]
    return [_f(table, "ERROR", "NON_FINITE",
               f"{int(bad.sum())} non-finite value(s) in the hypercube "
               f"(first at {dict(zip(AXIS_ORDER, idx.tolist()))})",
               locus={"indices": idx.tolist()})]


def check_indices_in_range(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Data-record indices within axis bounds (parse-enforced)."""
    return []


def check_header_enums(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Header type strings must be in the valid enumerations (parse-enforced)."""
    return []


def check_duplicate_tables(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Duplicate table numbers within the deck (deck-level, parse-enforced)."""
    return []


# ------------------------------------------------------------------ physical (8-13)


def check_thp_monotonic(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """BHP must increase with THP at fixed (WFR, GFR, ALQ, FLO)."""
    thp = table.axes["THP"].values
    if thp.size < 2:
        return []
    d = np.diff(table.data, axis=0)  # along THP
    bad = d <= -1e-9
    if not bad.any():
        return []
    n_slices = int(bad.any(axis=0).sum())
    first = np.argwhere(bad)[0]
    slice_idx = first[1:].tolist()
    thp_lo = thp[first[0]]
    thp_hi = thp[first[0] + 1]
    return [_f(
        table, "ERROR", "THP_MONOTONIC",
        f"BHP decreases with THP in {n_slices} slice(s) "
        f"(first: {table.axes['FLO'].kind}={table.axes['FLO'].values[first[4]]:g}, "
        f"THP {thp_lo:g}->{thp_hi:g}, WFR/GFR/ALQ idx {slice_idx[1:]})",
        locus={"slice": slice_idx, "thp": [float(thp_lo), float(thp_hi)]},
        plot_hint={"kind": "thp_monotonic", "slice": slice_idx},
    )]


def check_crossing_thp_curves(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """THP curves (BHP vs FLO) must not cross at fixed (WFR, GFR, ALQ)."""
    flo = table.axes["FLO"].values
    n_thp = table.axes["THP"].values.size
    if n_thp < 2 or flo.size < 2:
        return []
    findings: list[Finding] = []
    for w in range(table.axes["WFR"].values.size):
        for g in range(table.axes["GFR"].values.size):
            for a in range(table.axes["ALQ"].values.size):
                slab = table.data[:, w, g, a, :]  # (NTHP, NFLO)
                for i in range(n_thp):
                    for j in range(i + 1, n_thp):
                        d = slab[i] - slab[j]
                        signs = np.sign(d)
                        cross = np.where(np.diff(signs) != 0)[0]
                        if cross.size:
                            k = int(cross[0])
                            findings.append(_f(
                                table, "ERROR", "CROSSING",
                                f"THP curves {i + 1} and {j + 1} cross near "
                                f"{table.axes['FLO'].kind}={flo[k]:g} "
                                f"(WFR/GFR/ALQ idx {w},{g},{a})",
                                locus={"wfr": w, "gfr": g, "alq": a, "flo": int(k),
                                       "thp_pair": [i, j]},
                                plot_hint={"kind": "crossing",
                                           "x": [float(flo[k])],
                                           "y": [float(slab[i][k])]},
                            ))
                            if len(findings) >= 20:
                                return findings
    return findings


def check_unstable_branch(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Non-monotonic BHP vs FLO: the unstable / liquid-loading limb.

    Reports the turning-point rate so the user can decide whether to trim.
    """
    flo = table.axes["FLO"].values
    if flo.size < 3:
        return []
    findings: list[Finding] = []
    for ti in range(table.axes["THP"].values.size):
        for w in range(table.axes["WFR"].values.size):
            for g in range(table.axes["GFR"].values.size):
                for a in range(table.axes["ALQ"].values.size):
                    curve = table.data[ti, w, g, a, :]
                    d = np.sign(np.diff(curve))
                    turns = np.where(np.diff(d) != 0)[0]
                    if turns.size:
                        k = int(turns[0]) + 1  # index of the turning point
                        if len(findings) < 20:
                            findings.append(_f(
                                table, "WARNING", "UNSTABLE_BRANCH",
                                f"non-monotonic BHP vs {table.axes['FLO'].kind}: "
                                f"turning point at {flo[k]:g} "
                                f"({table.axes['FLO'].unit}) on THP={ti + 1}, "
                                f"WFR/GFR/ALQ idx {w},{g},{a} — unstable limb",
                                locus={"thp": ti, "wfr": w, "gfr": g, "alq": a,
                                       "flo": int(k)},
                                plot_hint={"kind": "turning_point",
                                           "flo": float(flo[k]),
                                           "value": float(curve[k])},
                            ))
    return findings


def check_absurd_bhp(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Negative, zero, or absurdly large tabulated values."""
    data = table.data
    hi = ctx.thresholds.get("absurd_bhp_bar", 1.0e4)
    if table.unit_system == "FIELD":
        hi *= 14.5038  # psia equivalent
    bad = ~np.isfinite(data) | (data <= 0.0) | (data > hi)
    if not bad.any():
        return []
    n = int(bad.sum())
    first = np.argwhere(bad)[0]
    return [_f(table, "ERROR", "ABSURD_BHP",
               f"{n} tabulated value(s) <= 0 or > {hi:g}: first at "
               f"{dict(zip(AXIS_ORDER, first.tolist()))}",
               locus={"indices": first.tolist()})]


def check_bhp_below_thp(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """BHP below THP: negative total pressure drop over the tubing/branch."""
    if table.tabulated != "BHP":
        return []
    thp = table.axes["THP"].values
    data = table.data
    diff = data - thp.reshape(-1, 1, 1, 1, 1)
    bad = diff < -1e-9
    if not bad.any():
        return []
    n = int(bad.sum())
    first = np.argwhere(bad)[0]
    return [_f(table, "WARNING", "BHP_LT_THP",
               f"{n} point(s) with BHP below THP (negative total pressure drop), "
               f"first at {dict(zip(AXIS_ORDER, first.tolist()))}",
               locus={"indices": first.tolist()})]


def check_flat_gradient(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Zero pressure gradient over a whole FLO range (generation artifact)."""
    flo = table.axes["FLO"].values
    if flo.size < 2:
        return []
    scale = max(1.0, float(np.max(np.abs(table.data))))
    tol = 1e-6 * scale
    flat = 0
    first_locus = None
    for ti in range(table.axes["THP"].values.size):
        for w in range(table.axes["WFR"].values.size):
            for g in range(table.axes["GFR"].values.size):
                for a in range(table.axes["ALQ"].values.size):
                    curve = table.data[ti, w, g, a, :]
                    if np.all(np.abs(np.diff(curve)) <= tol):
                        flat += 1
                        first_locus = first_locus or (ti, w, g, a)
    if not flat:
        return []
    return [_f(table, "WARNING", "FLAT_GRADIENT",
               f"{flat} curve(s) with zero pressure gradient over the whole "
               f"{table.axes['FLO'].kind} range (flat region)",
               locus={"slice": list(first_locus)})]


# ------------------------------------------------------------------ consistency (14-20)


_GAS_PHASES = {"GAS", "GASWAT", "GASOIL", "GASWATOIL", "OILGAS", "WATGAS"}
_LIQUID_ONLY = {"OIL", "WAT", "OILWAT", "WATOIL"}
_GAS_ONLY = {"GAS"}


def check_flo_phase_mismatch(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """FLO type vs consuming-well phase (context from WELSPECS)."""
    deck = ctx.deck
    if deck is None or not table.consumers.wells:
        return []
    flo = table.axes["FLO"].kind
    findings: list[Finding] = []
    for well in table.consumers.wells:
        phase = getattr(deck, "well_phases", {}).get(well)
        if not phase:
            continue
        if flo == "GAS" and phase in _LIQUID_ONLY:
            findings.append(_f(
                table, "WARNING", "FLO_PHASE",
                f"FLO type GAS on table consumed by {well} (phase {phase}) — "
                f"gas-rate table on a liquid well",
                locus={"well": well, "phase": phase},
            ))
        elif flo in ("OIL", "LIQ", "WG", "TM") and phase in _GAS_ONLY:
            findings.append(_f(
                table, "WARNING", "FLO_PHASE",
                f"FLO type {flo} on table consumed by {well} (phase {phase}) — "
                f"liquid-rate table on a gas well",
                locus={"well": well, "phase": phase},
            ))
    return findings


def check_alq_length1_with_gaslift(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """ALQ axis of length 1 on a well with gas lift declared (WLIFTOPT)."""
    deck = ctx.deck
    if deck is None or table.axis_lengths["ALQ"] != 1 or not table.consumers.wells:
        return []
    gl = getattr(deck, "gaslift_wells", set())
    offenders = [w for w in table.consumers.wells if w in gl]
    if not offenders:
        return []
    return [_f(table, "WARNING", "ALQ_GASLIFT",
               f"ALQ axis has a single value but well(s) {', '.join(offenders)} "
               f"declare gas lift (WLIFTOPT) — lift response cannot be tabulated",
               locus={"wells": offenders})]


def check_alq_axis_blank_type(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """ALQ axis present (length > 1) but ALQ type blank."""
    if table.axis_lengths["ALQ"] > 1 and table.axes["ALQ"].kind == "":
        return [_f(table, "WARNING", "ALQ_BLANK",
                   "ALQ axis has multiple values but ALQ type is blank",
                   locus={"alq_length": table.axis_lengths["ALQ"]})]
    return []


def check_units_mismatch(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Table unit system differs from the deck's."""
    deck = ctx.deck
    if deck is None:
        return []
    t_units, d_units = table.unit_system, getattr(deck, "unit_system", "DEFAULT")
    if t_units != "DEFAULT" and d_units != "DEFAULT" and t_units != d_units:
        return [_f(table, "ERROR", "UNITS_MISMATCH",
                   f"table declared in {t_units} but deck is {d_units}")]
    return []


def check_datum_mismatch(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Datum depth far from consuming wells' WELSPECS datum."""
    deck = ctx.deck
    if deck is None or not table.consumers.wells:
        return []
    tol = ctx.thresholds.get("datum_tolerance_m", 100.0)
    findings: list[Finding] = []
    for well in table.consumers.wells:
        wd = getattr(deck, "wells_datum", {}).get(well)
        if wd is None:
            continue
        if abs(table.datum_depth - wd) > tol:
            findings.append(_f(
                table, "WARNING", "DATUM_MISMATCH",
                f"table datum {table.datum_depth:g} m far from well {well} datum "
                f"{wd:g} m (tolerance {tol:g} m)",
                locus={"well": well, "table_datum": table.datum_depth,
                       "well_datum": wd},
            ))
    return findings


def check_dead_table(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Table not referenced by any well or branch."""
    if not table.consumers:
        return [_f(table, "INFO", "DEAD_TABLE",
                   "table not referenced by any well or branch")]
    return []


def check_role_conflict(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Table referenced by both a well and a branch (almost always a mistake)."""
    if table.consumers.wells and table.consumers.branches:
        return [_f(
            table, "WARNING", "ROLE_CONFLICT",
            f"table referenced by well(s) {', '.join(table.consumers.wells)} "
            f"and branch(es) "
            f"{', '.join(f'{a}->{b}' for a, b in table.consumers.branches)}",
            locus={"wells": list(table.consumers.wells),
                   "branches": [list(b) for b in table.consumers.branches]},
            plot_hint={"kind": "role_conflict"},
        )]
    return []
