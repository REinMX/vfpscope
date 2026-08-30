"""Interpolation tests: hand-computed multilinear values and clamp semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfpscope.core.interp import lookup
from vfpscope.core.parse.native import parse_vfp_file

FIXTURES = Path(__file__).parent / "fixtures"

# analytic formula of the synthetic fixture
def bhp(flo, thp, wct, gor, alq):
    return thp + 50.0 + 0.5 * flo + 20.0 * wct + 0.05 * gor + 0.001 * alq


@pytest.fixture(scope="module")
def table():
    return parse_vfp_file(FIXTURES / "synthetic_2x3x2x2x2.inc")


def test_interior_point_along_flo(table):
    r = lookup(table, flo=15.0, thp=100.0, wfr=0.0, gfr=100.0, alq=0.0)
    assert r.value == pytest.approx(bhp(15.0, 100.0, 0.0, 100.0, 0.0))
    assert r.clamped == {"THP": None, "WFR": None, "GFR": None, "ALQ": None, "FLO": None}


def test_interior_two_axis_point(table):
    r = lookup(table, flo=15.0, thp=250.0, wfr=0.0, gfr=100.0, alq=0.0)
    # linear in both FLO and THP -> analytic formula holds exactly
    assert r.value == pytest.approx(bhp(15.0, 250.0, 0.0, 100.0, 0.0))


def test_full_five_axis_interior_point(table):
    r = lookup(table, flo=15.0, thp=250.0, wfr=0.25, gfr=150.0, alq=25000.0)
    assert r.value == pytest.approx(bhp(15.0, 250.0, 0.25, 150.0, 25000.0))


def test_clamping_low_is_edge_value_not_extrapolation(table):
    r = lookup(table, flo=5.0, thp=100.0, wfr=0.0, gfr=100.0, alq=0.0)
    # clamped to FLO=10: value equals the table edge, NOT the linear extrapolation
    assert r.value == pytest.approx(bhp(10.0, 100.0, 0.0, 100.0, 0.0))
    assert r.clamped["FLO"] == "low"
    assert r.any_clamped


def test_clamping_high(table):
    r = lookup(table, flo=50.0, thp=100.0, wfr=0.0, gfr=100.0, alq=0.0)
    assert r.value == pytest.approx(bhp(20.0, 100.0, 0.0, 100.0, 0.0))
    assert r.clamped["FLO"] == "high"


def test_corner_clamp_all_axes(table):
    r = lookup(table, flo=999.0, thp=999.0, wfr=9.0, gfr=999.0, alq=1e9)
    assert r.value == pytest.approx(bhp(20.0, 300.0, 0.5, 200.0, 50000.0))
    assert all(v == "high" for v in r.clamped.values())


def test_exact_node_returns_tabulated_value(table):
    r = lookup(table, flo=10.0, thp=200.0, wfr=0.5, gfr=200.0, alq=0.0)
    assert r.value == pytest.approx(bhp(10.0, 200.0, 0.5, 200.0, 0.0))
    assert not r.any_clamped


def test_array_query_vectorized(table):
    flo = np.array([5.0, 10.0, 15.0, 20.0, 50.0])
    thp = np.full(5, 100.0)
    r = lookup(table, flo=flo, thp=thp, wfr=0.0, gfr=100.0, alq=0.0)
    expect = np.array([bhp(10.0, 100.0, 0.0, 100.0, 0.0),  # clamped low
                       bhp(10.0, 100.0, 0.0, 100.0, 0.0),
                       bhp(15.0, 100.0, 0.0, 100.0, 0.0),
                       bhp(20.0, 100.0, 0.0, 100.0, 0.0),
                       bhp(20.0, 100.0, 0.0, 100.0, 0.0)])  # clamped high
    np.testing.assert_allclose(r.value, expect)
    lo, hi = r.clamped_masks["FLO"]
    assert lo.tolist() == [True, False, False, False, False]
    assert hi.tolist() == [False, False, False, False, True]
    assert r.any_clamped


def test_array_mixed_axis_masks(table):
    r = lookup(table, flo=np.array([15.0, 15.0]), thp=np.array([100.0, 500.0]),
               wfr=0.0, gfr=100.0, alq=0.0)
    lo, hi = r.clamped_masks["THP"]
    assert hi.tolist() == [False, True]


def test_vfpinj_degenerate_axes_interpolate(table=None):
    inj = parse_vfp_file(FIXTURES / "synthetic_vfpinj_3x3.inc")
    # WFR/GFR/ALQ are length-1: any query value clamps to the single entry
    r = lookup(inj, flo=150.0, thp=100.0, wfr=0.3, gfr=123.0, alq=9.0)
    assert r.value == pytest.approx(100.0 + 40.0 + 0.2 * 150.0)
    assert r.clamped["WFR"] == "high"
    assert r.clamped["ALQ"] == "high"
    # on-value degenerate axis is not clamped
    r2 = lookup(inj, flo=150.0, thp=100.0, wfr=0.0, gfr=100.0, alq=0.0)
    assert r2.clamped["WFR"] is None


def test_mismatched_array_lengths_raise(table):
    with pytest.raises(ValueError):
        lookup(table, flo=np.array([1.0, 2.0]), thp=np.array([1.0, 2.0, 3.0]))
