"""QC engine: runs the registry over a deck, merges parse findings, dedupes."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Finding, VfpTable
from .checks import QCContext, check_absurd_bhp, check_alq_axis_blank_type
from .checks import check_alq_length1_with_gaslift, check_axes_monotonic
from .checks import check_bhp_below_thp, check_crossing_thp_curves, check_datum_mismatch
from .checks import check_dead_table, check_duplicate_tables, check_flat_gradient
from .checks import check_flo_phase_mismatch, check_header_enums, check_indices_in_range
from .checks import check_non_finite, check_record_count, check_record_width
from .checks import check_role_conflict, check_thp_monotonic, check_units_mismatch
from .checks import check_unstable_branch


@dataclass(frozen=True)
class CheckDef:
    check_id: str
    name: str
    severity: str
    description: str
    fn: object  # callable(table, ctx) -> list[Finding]


CHECK_REGISTRY: tuple[CheckDef, ...] = (
    CheckDef("RECORD_COUNT", "record count", "ERROR",
             "data record count != NTHP x NWFR x NGFR x NALQ (parse-enforced)", check_record_count),
    CheckDef("RECORD_WIDTH", "record width", "ERROR",
             "data record with != NFLO values (parse-enforced)", check_record_width),
    CheckDef("AXIS_MONOTONIC", "axis monotonic", "ERROR",
             "axis values not strictly increasing (parse-enforced)", check_axes_monotonic),
    CheckDef("NON_FINITE", "non-finite values", "ERROR",
             "NaN / inf entries in the hypercube", check_non_finite),
    CheckDef("INDEX_RANGE", "index range", "ERROR",
             "data-record index out of range (parse-enforced)", check_indices_in_range),
    CheckDef("HEADER_ENUM", "header enums", "ERROR",
             "header type strings outside valid enumerations", check_header_enums),
    CheckDef("DUP_TABLE", "duplicate tables", "ERROR",
             "duplicate table numbers in the deck", check_duplicate_tables),
    CheckDef("THP_MONOTONIC", "BHP vs THP monotonic", "ERROR",
             "BHP must increase with THP at fixed other axes", check_thp_monotonic),
    CheckDef("CROSSING", "crossing THP curves", "ERROR",
             "THP curves cross at fixed other axes", check_crossing_thp_curves),
    CheckDef("UNSTABLE_BRANCH", "unstable branch", "WARNING",
             "non-monotonic BHP vs FLO (liquid-loading limb)", check_unstable_branch),
    CheckDef("ABSURD_BHP", "absurd BHP", "ERROR",
             "negative, zero or absurdly large tabulated values", check_absurd_bhp),
    CheckDef("BHP_LT_THP", "BHP below THP", "WARNING",
             "negative total pressure drop over tubing/branch", check_bhp_below_thp),
    CheckDef("FLAT_GRADIENT", "flat gradient", "WARNING",
             "zero pressure gradient over a whole FLO range", check_flat_gradient),
    CheckDef("FLO_PHASE", "FLO vs fluid system", "WARNING",
             "FLO type contradicts consuming well phase", check_flo_phase_mismatch),
    CheckDef("ALQ_GASLIFT", "ALQ vs gas lift", "WARNING",
             "ALQ axis length 1 on a well with WLIFTOPT", check_alq_length1_with_gaslift),
    CheckDef("ALQ_BLANK", "blank ALQ type", "WARNING",
             "ALQ axis present but ALQ type blank", check_alq_axis_blank_type),
    CheckDef("UNITS_MISMATCH", "unit system", "ERROR",
             "table unit system differs from the deck's", check_units_mismatch),
    CheckDef("DATUM_MISMATCH", "datum depth", "WARNING",
             "datum far from consuming well's WELSPECS datum", check_datum_mismatch),
    CheckDef("DEAD_TABLE", "dead table", "INFO",
             "table not referenced by any well or branch", check_dead_table),
    CheckDef("ROLE_CONFLICT", "well/branch conflict", "WARNING",
             "table referenced by both a well and a branch", check_role_conflict),
)

_CHECK_BY_ID = {c.check_id: c for c in CHECK_REGISTRY}


def check_summary(
    findings: list[Finding],
) -> tuple[int, int, int]:
    """(n_error, n_warning, n_info) counts."""
    n_err = sum(1 for f in findings if f.severity == "ERROR")
    n_warn = sum(1 for f in findings if f.severity == "WARNING")
    n_info = sum(1 for f in findings if f.severity == "INFO")
    return n_err, n_warn, n_info


def run_qc(deck, context: QCContext | None = None) -> list[Finding]:
    """All findings for a deck: parse/ref findings + every registry check.

    Deduped by (check_id, table_number, message); deterministic order.
    """
    ctx = context or QCContext(deck=deck)
    out: list[Finding] = list(deck.findings)
    seen = {(f.check_id, f.table_number, f.message) for f in out}
    for no in deck.table_order:
        table: VfpTable = deck.tables[no]
        for check in CHECK_REGISTRY:
            for f in check.fn(table, ctx):  # type: ignore[attr-defined]
                key = (f.check_id, f.table_number, f.message)
                if key not in seen:
                    seen.add(key)
                    out.append(f)
    out.sort(key=lambda f: (f.table_number, f.check_id))
    return out


def worst_severity(findings: list[Finding]) -> str | None:
    order = {"INFO": 0, "WARNING": 1, "ERROR": 2}
    if not findings:
        return None
    return max((f.severity for f in findings), key=lambda s: order[s])
