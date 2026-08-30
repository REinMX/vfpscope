"""VFPPROD parsing tests: synthetic 2x3x2x2x2, real corpus files, error cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfpscope.core.model import AXIS_ORDER
from vfpscope.core.parse.native import VfpParseError, load_deck, parse_vfp_file

FIXTURES = Path(__file__).parent / "fixtures"

# analytic formula embedded in the fixture
def expected_bhp(flo, thp, wct, gor, alq):
    return thp + 50.0 + 0.5 * flo + 20.0 * wct + 0.05 * gor + 0.001 * alq


def test_parses_synthetic_2x3x2x2x2():
    t = parse_vfp_file(FIXTURES / "synthetic_2x3x2x2x2.inc")
    assert t.number == 1
    assert t.kind == "VFPPROD"
    assert t.datum_depth == 1500.0
    assert t.unit_system == "METRIC"
    assert t.tabulated == "BHP"
    assert t.axes["FLO"].kind == "LIQ"
    assert t.axes["WFR"].kind == "WCT"
    assert t.axes["GFR"].kind == "GOR"
    assert t.axes["ALQ"].kind == "GRAT"
    assert t.data.shape == (3, 2, 2, 2, 2)


def test_every_value_matches_analytic_fixture():
    t = parse_vfp_file(FIXTURES / "synthetic_2x3x2x2x2.inc")
    flo = t.axes["FLO"].values
    thp = t.axes["THP"].values
    wct = t.axes["WFR"].values
    gor = t.axes["GFR"].values
    alq = t.axes["ALQ"].values
    for ti, tv in enumerate(thp):
        for wi, wv in enumerate(wct):
            for gi, gv in enumerate(gor):
                for ai, av in enumerate(alq):
                    expect = np.array(
                        [expected_bhp(f, tv, wv, gv, av) for f in flo]
                    )
                    np.testing.assert_allclose(t.data[ti, wi, gi, ai, :], expect)


def test_axis_units_resolved_metric():
    t = parse_vfp_file(FIXTURES / "synthetic_2x3x2x2x2.inc")
    assert t.axes["FLO"].unit == "sm3/day"
    assert t.axes["THP"].unit == "barsa"
    assert t.axes["ALQ"].unit == "sm3/day"


def test_record_count_mismatch_raises_located_error():
    with pytest.raises(VfpParseError) as ei:
        parse_vfp_file(FIXTURES / "broken_count_23of24.inc")
    msg = str(ei.value)
    assert "expected 24 data records, found 23" in msg
    assert "broken_count_23of24.inc" in msg


def test_short_data_row_raises():
    with pytest.raises(VfpParseError) as ei:
        parse_vfp_file(FIXTURES / "broken_width_shortrow.inc")
    msg = str(ei.value)
    assert "expected 2 value(s), found 1" in msg
    assert "broken_width_shortrow.inc" in msg


def test_out_of_range_index_raises():
    with pytest.raises(VfpParseError) as ei:
        parse_vfp_file(FIXTURES / "broken_index_4of3.inc")
    msg = str(ei.value)
    assert "THP" in msg and "4" in msg


def test_parses_real_master_tables_three_roles_unknown():
    # vfp_tables_master.inc: 2 well tables (datum 2000) + 1 trunk table (datum 0)
    deck = load_deck("/home/javier/worktrees/oil-gas-priority-toy/spikes/004-opm-flow-master/vfp_tables_master.inc")
    assert sorted(deck.tables) == [1, 2, 3]
    t3 = deck.tables[3]
    assert t3.datum_depth == 0.0
    assert t3.data.shape == (5, 2, 2, 1, 5)


def test_duplicate_table_numbers_are_findings():
    text = open(FIXTURES / "synthetic_2x3x2x2x2.inc").read()
    dup = text + "\n" + text.replace("VFPPROD\n  1 ", "VFPPROD\n  1 ", 1)
    p = FIXTURES / "_tmp_dup.inc"
    p.write_text(dup)
    try:
        deck = load_deck(p)
        assert len(deck.tables) == 1
        assert any(f.check_id == "DUP_TABLE" for f in deck.findings)
    finally:
        p.unlink()
