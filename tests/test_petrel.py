"""Petrel simulation view: labels, operating point, run gate.

Eclipse linearly extrapolates outside a VFP table (Reference Manual, VFPPROD).
The clamped lookup is the table edge, not the Petrel/Eclipse answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vfpscope.core.parse.native import load_deck, parse_vfp_file
from vfpscope.core.petrel import (
    assignments_csv,
    axis_caption,
    evaluate_operating_point,
    finding_implication,
    quantity_caption,
    simulation_brief,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def prod():
    return parse_vfp_file(FIXTURES / "synthetic_2x3x2x2x2.inc")


@pytest.fixture(scope="module")
def inj():
    return parse_vfp_file(FIXTURES / "synthetic_vfpinj_3x3.inc")


def test_axis_caption_uses_petrel_names_and_units(prod):
    assert axis_caption(prod, "FLO") == "Liquid rate (sm3/day)"
    assert axis_caption(prod, "WFR") == "Water cut (sm3/sm3)"
    assert axis_caption(prod, "GFR") == "Gas-oil ratio (sm3/sm3)"
    assert axis_caption(prod, "ALQ") == "Gas-lift rate (sm3/day)"
    assert axis_caption(prod, "THP") == "Tubing-head pressure (barsa)"


def test_branch_table_relabels_inlet_and_outlet(prod):
    branch = prod.model_copy(update={"role": "BRANCH"})
    assert axis_caption(branch, "THP") == "Outlet pressure (barsa)"
    assert quantity_caption(branch) == "Inlet pressure"


def test_interior_point_matches_table_and_says_inside(prod):
    # BHP = THP + 50 + 0.5*FLO + 20*WCT + 0.05*GOR + 0.001*ALQ
    point = evaluate_operating_point(
        prod, flo=15.0, thp=100.0, wfr=0.0, gfr=100.0, alq=0.0
    )
    assert point.inside is True
    assert point.linear_bhp == pytest.approx(162.5)
    assert point.edge_bhp == pytest.approx(162.5)
    assert "inside" in point.note.lower()
    assert point.axis("FLO").status == "inside"


def test_outside_rate_extrapolates_edge_segment_and_does_not_claim_eclipse_run(prod):
    point = evaluate_operating_point(
        prod, flo=30.0, thp=100.0, wfr=0.0, gfr=100.0, alq=0.0
    )
    assert point.inside is False
    assert point.axis("FLO").status == "high"
    assert point.edge_bhp == pytest.approx(165.0)  # table edge at FLO=20
    assert point.linear_bhp == pytest.approx(170.0)  # continue 0.5 bar per sm3/day
    note = point.note.lower()
    assert "extrapolat" in note
    assert "not an eclipse run" in note


def test_length1_axis_is_independent_not_extrapolated(inj):
    point = evaluate_operating_point(inj, flo=200.0, thp=100.0, wfr=5.0)
    assert point.inside is True
    assert point.axis("WFR").status == "independent"
    # BHP = THP + 40 + 0.2*FLO, independent of the singleton water axis
    assert point.linear_bhp == pytest.approx(180.0)
    assert point.edge_bhp == pytest.approx(180.0)


def test_single_thp_moves_bhp_one_for_one(tmp_path):
    path = tmp_path / "one_thp.inc"
    path.write_text(
        "VFPPROD\n"
        " 1 1000.0 'LIQ' 'WCT' 'GOR' 'THP' 'GRAT' 'METRIC' 'BHP' /\n"
        " 10 20 /\n 100 /\n 0.0 /\n 100 /\n 0.0 /\n"
        " 1 1 1 1 160.0 170.0 /\n/\n"
    )
    table = parse_vfp_file(path)
    point = evaluate_operating_point(table, flo=20.0, thp=130.0, wfr=0.0, gfr=100.0, alq=0.0)
    assert point.axis("THP").status == "unit_slope"
    assert point.linear_bhp == pytest.approx(200.0)
    assert point.edge_bhp == pytest.approx(170.0)
    assert "not an eclipse run" in point.note.lower()
    on_node = evaluate_operating_point(table, flo=20.0, thp=100.0, wfr=0.0, gfr=100.0, alq=0.0)
    assert on_node.linear_bhp == pytest.approx(170.0)
    assert on_node.axis("THP").status == "inside"


def test_temperature_table_uses_temperature_units_not_pressure(tmp_path):
    path = tmp_path / "tht.inc"
    path.write_text(
        "VFPPROD\n"
        " 1 1000.0 'LIQ' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'TEMP' /\n"
        " 10 20 /\n 100 /\n 0.0 /\n 100 /\n 0.0 /\n"
        " 1 1 1 1 40.0 45.0 /\n/\n"
    )
    table = parse_vfp_file(path)
    point = evaluate_operating_point(table, flo=10.0, thp=100.0)
    assert point.quantity == "Tubing-head temperature"
    assert point.unit == "°C"
    shifted = evaluate_operating_point(table, flo=10.0, thp=120.0)
    assert shifted.linear_bhp == pytest.approx(40.0)
    field = table.model_copy(update={"unit_system": "FIELD"})
    assert evaluate_operating_point(field, flo=10.0, thp=100.0).unit == "°F"


def test_clean_assigned_table_is_ready_to_run(tmp_path):
    deckf = tmp_path / "case.DATA"
    deckf.write_text(
        "METRIC\n"
        "WELSPECS\n 'P1' 'G1' 1 1 1000.0 'OIL' /\n/\n"
        "WCONPROD\n 'P1' OPEN ORAT 1* 1* 1* 1* 1* 1* 1* 1 /\n/\n"
        "VFPPROD\n 1 1000.0 'LIQ' 'WCT' 'GOR' 'THP' 'GRAT' 'METRIC' 'BHP' /\n"
        " 10 20 /\n 100 200 /\n 0.0 /\n 100 /\n 0.0 /\n"
        " 1 1 1 1 160.0 170.0 /\n"
        " 2 1 1 1 260.0 270.0 /\n/\n"
    )
    brief = simulation_brief(load_deck(deckf))
    assert brief.gate == "RUN"
    assert brief.unit_system == "METRIC"
    assert brief.wells[0].well == "P1"
    assert brief.wells[0].table_number == 1
    assert brief.wells[0].table_present is True
    assert "No blocking" in brief.headline


def test_missing_table_blocks_the_petrel_run(tmp_path):
    deckf = tmp_path / "case.DATA"
    deckf.write_text(
        "WCONPROD\n 'P1' OPEN ORAT 1* 1* 1* 1* 1* 1* 1* 99 /\n/\n"
        "VFPPROD\n 1 1000.0 'LIQ' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'BHP' /\n"
        " 10 20 /\n 100 /\n 0.0 /\n 100 /\n 0.0 /\n"
        " 1 1 1 1 160.0 165.0 /\n/\n"
    )
    brief = simulation_brief(load_deck(deckf))
    assert brief.gate == "DO_NOT_RUN"
    assert any(w.well == "P1" and w.table_number == 99 and not w.table_present for w in brief.wells)
    assert brief.headline.startswith("Do not run")


def test_norne_blocks_run_and_lists_unassigned_table():
    brief = simulation_brief(load_deck(FIXTURES / "norne_vfp_deck.DATA"))
    assert brief.gate == "DO_NOT_RUN"
    assert 6 in brief.unassigned_tables
    assert any(w.well == "B1" and w.table_number == 1 for w in brief.wells)


def test_assignments_csv_is_inspectable():
    brief = simulation_brief(load_deck(FIXTURES / "norne_vfp_deck.DATA"))
    csv_text = assignments_csv(brief)
    header = csv_text.splitlines()[0]
    assert header == "well,phase,datum,vfp_table,table_present,table_kind,table_datum,gas_lift"
    assert "B1," in csv_text


def test_unstable_branch_implication_names_convergence():
    text = finding_implication("UNSTABLE_BRANCH")
    assert "convergence" in text.lower() or "oscillat" in text.lower()
