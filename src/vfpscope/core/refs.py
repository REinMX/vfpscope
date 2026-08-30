"""Reference mapping: WCONPROD / WCONINJE / BRANPROP / NODEPROP / WELSPECS
scans turned into per-table role and consumer assignments.

Role inference rules (spec 3.3):

* table number appears in ``BRANPROP``          -> BRANCH
* table number appears in ``WCONPROD``/``WCONINJE`` -> WELL
* both                                            -> WELL + ROLE_CONFLICT finding
* neither                                          -> UNKNOWN (user may set manually)
"""

from __future__ import annotations

from .model import Consumers

# well phase kinds that are "gas-like" / "liquid-like" for the FLO-type
# consistency check (check 14)
GAS_PHASES = {"GAS", "GASWAT", "GASOIL", "GASWATOIL", "OILGAS", "WATgas".upper()}
LIQUID_PHASES = {"OIL", "WAT", "OILWAT", "WATOIL"}


def assign_roles(deck) -> None:
    """Infer ``role`` and ``consumers`` for every table in the deck.

    Tables are immutable, so each is replaced by a ``model_copy`` with the
    assigned role/consumers.
    """
    for no in deck.table_order:
        t = deck.tables[no]
        wells = tuple(sorted({w for w, v in deck.well_vfp.items() if v == no}))
        branches = tuple(sorted({(dt, ut) for (dt, ut), v in deck.branch_vfp.items() if v == no}))
        consumers = Consumers(wells=wells, branches=branches)
        if wells and branches:
            role = "WELL"  # conflict finding comes from QC check ROLE_CONFLICT
        elif wells:
            role = "WELL"
        elif branches:
            role = "BRANCH"
        else:
            role = "UNKNOWN"
        deck.tables[no] = t.model_copy(update={"role": role, "consumers": consumers})
    # dangling references: wells/branches pointing at tables that do not exist
    known = set(deck.tables)
    for well, no in deck.well_vfp.items():
        if no not in known:
            deck.finding(
                "MISSING_TABLE",
                "ERROR",
                no,
                f"well {well} references VFP table {no}, which is not defined in the deck",
                locus={"well": well},
            )
    for (dt, ut), no in deck.branch_vfp.items():
        if no not in known:
            deck.finding(
                "MISSING_TABLE",
                "ERROR",
                no,
                f"branch {dt}->{ut} references VFP table {no}, which is not "
                f"defined in the deck",
                locus={"branch": [dt, ut]},
            )


def referencing_wells(deck, table_no: int) -> tuple[str, ...]:
    return tuple(sorted({w for w, v in deck.well_vfp.items() if v == table_no}))


def referencing_branches(deck, table_no: int) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({(dt, ut) for (dt, ut), v in deck.branch_vfp.items() if v == table_no}))
