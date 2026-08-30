"""QC engine tests: physical/consistency checks on deliberately broken decks."""

from __future__ import annotations

from pathlib import Path

from vfpscope.core.parse.native import load_deck
from vfpscope.core.qc.engine import run_qc

FIXTURES = Path(__file__).parent / "fixtures"

# ------------------------------------------------------------------ helpers


def _deck_text(vfpprod_block: str, extra: str = "") -> str:
    return (
        "RUNSPEC\nMETRIC\nSCHEDULE\n"
        + extra
        + "\nVFPPROD\n"
        + vfpprod_block
        + "\n/\n"
    )


SIMPLE_HEADER = " 1 1500.0 'LIQ' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'BHP' /\n"
SIMPLE_AXES = " 10 20 /\n 100 200 /\n 0.0 /\n 100 /\n 0.0 /\n"  # FLO THP WCT GOR ALQ


def _ids(deck, check_id: str) -> list[int]:
    return sorted({f.table_number for f in run_qc(deck) if f.check_id == check_id})


def _load_text(text: str, tmp_path: Path):
    p = tmp_path / "deck.DATA"
    p.write_text(text)
    return load_deck(p)


# ------------------------------------------------------------------ physical checks


def test_thp_non_monotonic(tmp_path):
    # THP=200 gives LOWER BHP than THP=100 at FLO=20 -> violation
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + SIMPLE_AXES
                   + " 1 1 1 1 160.0 165.0 /\n 2 1 1 1 170.0 155.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "THP_MONOTONIC"]
    assert len(fs) == 1
    assert fs[0].severity == "ERROR"
    # violation at FLO index 1: slice (WFR,GFR,ALQ,FLO) = (0,0,0,1)
    assert fs[0].plot_hint == {"kind": "thp_monotonic", "slice": [0, 0, 0, 1]}


def test_crossing_thp_curves(tmp_path):
    # curve THP=100 rises steeply and overtakes THP=200 -> crossing
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + " 10 20 30 /\n 100 200 /\n 0.0 /\n 100 /\n 0.0 /\n"
                   + " 1 1 1 1 160.0 240.0 300.0 /\n 2 1 1 1 260.0 250.0 240.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "CROSSING"]
    assert fs, "expected CROSSING finding"
    assert fs[0].severity == "ERROR"
    assert fs[0].plot_hint["kind"] == "crossing"


def test_unstable_branch_turning_point(tmp_path):
    # BHP dips at FLO=20 then recovers: unstable limb with turning point
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + " 10 20 30 /\n 100 /\n 0.0 /\n 100 /\n 0.0 /\n"
                   + " 1 1 1 1 160.0 155.0 165.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "UNSTABLE_BRANCH"]
    assert len(fs) == 1
    assert fs[0].severity == "WARNING"
    assert fs[0].plot_hint["kind"] == "turning_point"
    assert fs[0].plot_hint["flo"] == 20.0  # turning-point rate in the message/UI


def test_negative_bhp_is_error(tmp_path):
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + SIMPLE_AXES
                   + " 1 1 1 1 -5.0 165.0 /\n 2 1 1 1 260.0 265.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "ABSURD_BHP"]
    assert fs and fs[0].severity == "ERROR"


def test_absurdly_large_bhp_is_error(tmp_path):
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + SIMPLE_AXES
                   + " 1 1 1 1 1.0e5 165.0 /\n 2 1 1 1 260.0 265.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "ABSURD_BHP"]
    assert fs and fs[0].severity == "ERROR"


def test_bhp_below_thp_warning(tmp_path):
    # BHP 90 < THP 100 -> negative total pressure drop
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + SIMPLE_AXES
                   + " 1 1 1 1 90.0 95.0 /\n 2 1 1 1 260.0 265.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "BHP_LT_THP"]
    assert fs and fs[0].severity == "WARNING"


def test_flat_gradient_warning(tmp_path):
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + " 10 20 30 /\n 100 /\n 0.0 /\n 100 /\n 0.0 /\n"
                   + " 1 1 1 1 160.0 160.0 160.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "FLAT_GRADIENT"]
    assert fs and fs[0].severity == "WARNING"


# ------------------------------------------------------------------ consistency checks


def test_units_mismatch_is_error(tmp_path):
    block = SIMPLE_HEADER.replace("'METRIC'", "'FIELD'") + SIMPLE_AXES \
        + " 1 1 1 1 160.0 165.0 /\n 2 1 1 1 260.0 265.0 /\n"
    deck = _load_text(_deck_text(block), tmp_path)  # deck METRIC, table FIELD
    fs = [f for f in run_qc(deck) if f.check_id == "UNITS_MISMATCH"]
    assert fs and fs[0].severity == "ERROR"


def test_datum_mismatch_warning(tmp_path):
    deck = _load_text(
        _deck_text(
            SIMPLE_HEADER + SIMPLE_AXES
            + " 1 1 1 1 160.0 165.0 /\n 2 1 1 1 260.0 265.0 /\n",
            extra="WELSPECS\n 'P1' 'G1' 5 5 2000.0 'OIL' /\n/\n"
                  "WCONPROD\n 'P1' OPEN ORAT 1* 1* 1* 1* 1* 1* 1* 1 /\n/\n",
        ),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "DATUM_MISMATCH"]
    assert fs and fs[0].severity == "WARNING"


def test_alq_axis_blank_type_warning(tmp_path):
    # ALQ axis with 2 values but blank ALQ type
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + SIMPLE_AXES.replace(" 0.0 /\n 100 /\n 0.0 /\n",
                                                       " 0.0 /\n 100 /\n 0.0 50000.0 /\n")
                   + " 1 1 1 1 160.0 165.0 /\n 1 1 1 2 260.0 265.0 /\n"
                   + " 2 1 1 1 260.0 265.0 /\n 2 1 1 2 360.0 365.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "ALQ_BLANK"]
    assert fs and fs[0].severity == "WARNING"


def test_alq_length1_with_gaslift_warning(tmp_path):
    deck = _load_text(
        _deck_text(
            SIMPLE_HEADER + SIMPLE_AXES
            + " 1 1 1 1 160.0 165.0 /\n 2 1 1 1 260.0 265.0 /\n",
            extra="WLIFTOPT\n 'P1' 1.0 100.0 /\n/\n"
                  "WCONPROD\n 'P1' OPEN ORAT 1* 1* 1* 1* 1* 1* 1* 1 /\n/\n",
        ),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "ALQ_GASLIFT"]
    assert fs and fs[0].severity == "WARNING"


def test_flo_phase_mismatch_warning(tmp_path):
    # OIL well consuming a GAS-FLO table
    header = " 1 1500.0 'GAS' 'WGR' 'OGR' 'THP' ' ' 'METRIC' 'BHP' /\n"
    axes = " 10 20 /\n 100 200 /\n 0.0 /\n 100 /\n 0.0 /\n"
    deck = _load_text(
        _deck_text(
            header + axes + " 1 1 1 1 160.0 165.0 /\n 2 1 1 1 260.0 265.0 /\n",
            extra="WELSPECS\n 'P1' 'G1' 5 5 1500.0 'OIL' /\n/\n"
                  "WCONPROD\n 'P1' OPEN ORAT 1* 1* 1* 1* 1* 1* 1* 1 /\n/\n",
        ),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "FLO_PHASE"]
    assert fs and fs[0].severity == "WARNING"


def test_dead_table_info(tmp_path):
    deck = _load_text(
        _deck_text(SIMPLE_HEADER + SIMPLE_AXES
                   + " 1 1 1 1 160.0 165.0 /\n 2 1 1 1 260.0 265.0 /\n"),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "DEAD_TABLE"]
    assert fs and fs[0].severity == "INFO"


def test_role_conflict_warning(tmp_path):
    deck = _load_text(
        _deck_text(
            SIMPLE_HEADER + SIMPLE_AXES
            + " 1 1 1 1 160.0 165.0 /\n 2 1 1 1 260.0 265.0 /\n",
            extra="WCONPROD\n 'P1' OPEN ORAT 1* 1* 1* 1* 1* 1* 1* 1 /\n/\n"
                  "BRANPROP\n 'B1' 'FIELD' 1 /\n/\n",
        ),
        tmp_path,
    )
    fs = [f for f in run_qc(deck) if f.check_id == "ROLE_CONFLICT"]
    assert fs and fs[0].severity == "WARNING"


# ------------------------------------------------------------------ regression on real deck


def test_norne_deck_qc_regression():
    deck = load_deck(FIXTURES / "norne_vfp_deck.DATA")
    fs = run_qc(deck)
    by_id = {f.check_id: f for f in fs}
    # expected deliberate conditions in the fixture deck
    conflicts = {f.table_number for f in fs if f.check_id == "ROLE_CONFLICT"}
    assert conflicts == {8, 9}  # both branch tables also used by wells
    assert by_id["FLO_PHASE"].table_number == 5  # D1 (OIL) uses GAS table
    assert by_id["ALQ_GASLIFT"].table_number == 2  # E1 WLIFTOPT, ALQ len 1
    assert by_id["DEAD_TABLE"].table_number == 6
    # real findings in the Norne flowline table 9 (pe2.VFP): the QC engine
    # catches genuine corruption in the corpus — BHP dips when THP rises
    assert by_id["THP_MONOTONIC"].table_number == 9
    assert by_id["UNSTABLE_BRANCH"].table_number == 9
