"""VFPINJ parsing tests (2-axis case, real Norne file + synthetic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfpscope.core.parse.native import VfpParseError, parse_vfp_file

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_synthetic_vfpinj_3x3():
    t = parse_vfp_file(FIXTURES / "synthetic_vfpinj_3x3.inc")
    assert t.number == 7
    assert t.kind == "VFPINJ"
    assert t.datum_depth == 1650.0
    assert t.axes["FLO"].kind == "WAT"
    assert t.axes["THP"].values.tolist() == [50.0, 100.0, 150.0]
    assert t.data.shape == (3, 1, 1, 1, 3)
    # analytic: BHP = THP + 40 + 0.2*FLO
    for ti, tv in enumerate([50.0, 100.0, 150.0]):
        np.testing.assert_allclose(
            t.data[ti, 0, 0, 0, :], [tv + 40.0 + 0.2 * f for f in (100.0, 200.0, 300.0)]
        )


def test_vfpinj_degenerate_axes_are_length_one():
    t = parse_vfp_file(FIXTURES / "synthetic_vfpinj_3x3.inc")
    assert t.axes["WFR"].values.size == 1
    assert t.axes["GFR"].values.size == 1
    assert t.axes["ALQ"].values.size == 1


def test_parses_real_norne_vfpinj():
    p = FIXTURES / "norne_C1H.Ecl"
    t = parse_vfp_file(p)
    assert t.kind == "VFPINJ"
    assert t.axes["FLO"].kind == "WAT"
    # datum depth from real header
    assert t.datum_depth == pytest.approx(2718.07)
    assert t.axes["FLO"].values.size > 5
    assert t.axes["THP"].values.size > 1
    assert t.data.shape[0] == t.axes["THP"].values.size
    assert t.data.shape[-1] == t.axes["FLO"].values.size
    # all finite, positive
    assert np.isfinite(t.data).all()
    assert (t.data > 0).all()


def test_vfpinj_header_without_units_defaults():
    # 3-item header (table, datum, rate type) — units default to deck/DEFAULT
    p = FIXTURES / "norne_C1H.Ecl"
    t = parse_vfp_file(p)
    assert t.unit_system in ("METRIC", "FIELD", "DEFAULT")
