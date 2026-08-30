"""Reference mapping + role inference tests (M1 acceptance)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vfpscope.core.parse.native import VfpParseError, load_deck

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def norne_deck():
    return load_deck(FIXTURES / "norne_vfp_deck.DATA")


def test_all_tables_found(norne_deck):
    assert len(norne_deck.tables) == 10
    assert sorted(norne_deck.tables) == [1, 2, 3, 4, 5, 6, 8, 9, 12, 17]


def test_includes_resolved_from_deck_dir(norne_deck):
    t1 = norne_deck.tables[1]
    assert t1.source.path.endswith("norne/DevNew.VFP")
    assert t1.datum_depth == pytest.approx(2600.6)


def test_roles_inferred(norne_deck):
    roles = {no: t.role for no, t in norne_deck.tables.items()}
    assert roles[1] == "WELL"  # WCONPROD
    assert roles[2] == "WELL"
    assert roles[4] == "WELL"
    assert roles[6] == "UNKNOWN"  # dead table
    assert roles[8] == "WELL"  # well + branch -> WELL (conflict flagged)
    assert roles[9] == "WELL"
    assert roles[12] == "WELL"  # WCONINJE
    assert roles[17] == "WELL"


def test_consumers_mapping(norne_deck):
    t1 = norne_deck.tables[1]
    assert t1.consumers.wells == ("B1",)
    t8 = norne_deck.tables[8]
    assert t8.consumers.wells == ("PD2W",)
    assert t8.consumers.branches == (("FLS", "FIELD"),)
    t6 = norne_deck.tables[6]
    assert not t6.consumers


def test_well_branch_conflict_flagged(norne_deck):
    from vfpscope.core.qc.engine import run_qc

    fs = run_qc(norne_deck)
    assert any(f.check_id == "ROLE_CONFLICT" and f.table_number == 8 for f in fs)


def test_welspecs_and_gaslift_collected(norne_deck):
    assert norne_deck.wells_datum["B1"] == pytest.approx(2600.0)
    assert norne_deck.well_phases["D1"] == "OIL"
    assert "E1" in norne_deck.gaslift_wells
    assert ("FLS", "FIELD") in norne_deck.branch_vfp
    assert norne_deck.branch_vfp[("FLS", "FIELD")] == 8
    # NODEPROP nodes
    names = {n for n, _ in norne_deck.nodes}
    assert names == {"FIELD", "FLS", "FLN"}
    p_field = dict(norne_deck.nodes)["FIELD"]
    assert p_field == pytest.approx(80.0)


def test_deck_unit_system_detected(norne_deck):
    assert norne_deck.unit_system == "METRIC"


# ---------------------------------------------------------------- INCLUDE / PATHS


def test_paths_alias_resolution(tmp_path):
    inc = tmp_path / "tables.inc"
    inc.write_text(
        "VFPPROD\n 1 1000.0 'LIQ' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'BHP' /\n"
        " 10 20 /\n 100 /\n 0.0 /\n 100 /\n 0.0 /\n"
        " 1 1 1 1 160.0 165.0 /\n/\n"
    )
    deckf = tmp_path / "deck.DATA"
    deckf.write_text(
        f"PATHS\n 'VFPINC' '{tmp_path.as_posix()}/' /\n/\n"
        "INCLUDE\n '$VFPINC/tables.inc' /\n/\n"
    )
    deck = load_deck(deckf)
    assert 1 in deck.tables
    assert deck.tables[1].axes["FLO"].values.tolist() == [10.0, 20.0]


def test_unknown_alias_raises_clear_error(tmp_path):
    deckf = tmp_path / "deck.DATA"
    deckf.write_text("INCLUDE\n '$NOPE/tables.inc' /\n/\n")
    with pytest.raises(VfpParseError) as ei:
        load_deck(deckf)
    assert "unknown PATHS alias '$NOPE'" in str(ei.value)


def test_missing_include_raises_clear_error(tmp_path):
    deckf = tmp_path / "deck.DATA"
    deckf.write_text("INCLUDE\n 'missing.inc' /\n/\n")
    with pytest.raises(VfpParseError) as ei:
        load_deck(deckf)
    assert "include file not found" in str(ei.value)
    assert "missing.inc" in str(ei.value)


def test_include_cycle_detected(tmp_path):
    a = tmp_path / "a.inc"
    b = tmp_path / "b.inc"
    a.write_text("INCLUDE\n 'b.inc' /\n/\n")
    b.write_text("INCLUDE\n 'a.inc' /\n/\n")
    deckf = tmp_path / "deck.DATA"
    deckf.write_text("INCLUDE\n 'a.inc' /\n/\n")
    with pytest.raises(VfpParseError) as ei:
        load_deck(deckf)
    assert "cycle" in str(ei.value)


def test_missing_table_reference_is_finding(tmp_path):
    deckf = tmp_path / "deck.DATA"
    deckf.write_text(
        "WCONPROD\n 'P1' OPEN ORAT 1* 1* 1* 1* 1* 1* 1* 99 /\n/\n"
        "VFPPROD\n 1 1000.0 'LIQ' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'BHP' /\n"
        " 10 20 /\n 100 /\n 0.0 /\n 100 /\n 0.0 /\n"
        " 1 1 1 1 160.0 165.0 /\n/\n"
    )
    deck = load_deck(deckf)
    assert 1 in deck.tables
    assert any(
        f.check_id == "MISSING_TABLE" and f.table_number == 99 for f in deck.findings
    )
