"""M5/M6 tests: coverage overlay, compare guards, network graph, derive fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfpscope.core.coverage import CoverageReport, coverage_from_summary
from vfpscope.core.derive import dp_curve, fit_pressure_drop, gradient, turning_points
from vfpscope.core.parse.native import load_deck, parse_vfp_file
from vfpscope.viz.figures import (
    compare_difference_figure,
    compare_figure,
    coverage_overlay_figure,
    network_graph_figure,
)

FIXTURES = Path(__file__).parent / "fixtures"
TABLE_A = FIXTURES / "synthetic_2x3x2x2x2.inc"


class SummaryFake:
    def __init__(self, vectors):
        self._v = {k.upper(): v for k, v in vectors.items()}

    def numpy_vector(self, key):
        k = key.upper()
        if k not in self._v:
            raise KeyError(k)
        return self._v[k]


# ------------------------------------------------------------------ M5 overlay


def test_coverage_overlay_figure_builds():
    table = parse_vfp_file(TABLE_A)
    n = 100
    flo = np.linspace(5, 30, n)  # clamped on both sides (table 10..20)
    rep = CoverageReport(
        table_number=1, well="P1", n_timesteps=n,
        axis_values={"FLO": flo, "THP": np.full(n, 150.0)},
        clamped_low={"FLO": flo < 10, "THP": np.zeros(n, bool)},
        clamped_high={"FLO": flo > 20, "THP": np.zeros(n, bool)},
        bhp=200 + 0.5 * flo,
    )
    fig = coverage_overlay_figure(table, rep)
    assert len(fig.data) >= 4  # run scatter + 3 table curves + 3 bars
    assert any(tr.type == "bar" for tr in fig.data)
    assert "coverage" in fig.layout.title.text.lower()


# ------------------------------------------------------------------ M6 compare


def test_compare_figure_overlays_two_tables():
    t1 = parse_vfp_file(TABLE_A)
    t2 = parse_vfp_file(TABLE_A).model_copy(update={"number": 2})
    fig = compare_figure([t1, t2])
    assert len(fig.data) == 6  # 2 tables x 3 THP curves
    names = {tr.name for tr in fig.data}
    assert any(n.startswith("table 1") for n in names)
    assert any(n.startswith("table 2") for n in names)


def test_compare_refuses_unit_system_mismatch():
    t1 = parse_vfp_file(TABLE_A)  # METRIC
    t2 = t1.model_copy(update={"unit_system": "FIELD", "number": 2})
    with pytest.raises(ValueError, match="unit system"):
        compare_figure([t1, t2])


def test_compare_refuses_axis_type_mismatch():
    t1 = parse_vfp_file(TABLE_A)
    t2 = t1.model_copy(update={"number": 2})
    t2.axes["FLO"].kind  # noqa: B018
    # rebuild with different FLO kind

    t2 = t1.model_copy(update={"number": 2, "axes": t1.axes | {"FLO": t1.axes["FLO"].model_copy(update={"kind": "GAS"})}})
    with pytest.raises(ValueError, match="FLO types"):
        compare_figure([t1, t2])


def test_compare_difference_figure():
    t1 = parse_vfp_file(TABLE_A)
    t2 = t1.model_copy(update={"number": 2})
    fig = compare_difference_figure(t1, t2)
    assert len(fig.data) == min(3, 8)
    np.testing.assert_allclose(fig.data[0].y, 0.0, atol=1e-12)  # identical tables


def test_compare_difference_requires_equal_grid():
    t1 = parse_vfp_file(TABLE_A)
    t2 = parse_vfp_file(FIXTURES / "synthetic_vfpinj_3x3.inc")
    with pytest.raises(ValueError):
        compare_difference_figure(t1, t2)


# ------------------------------------------------------------------ M6 network


def test_network_graph_figure():
    deck = load_deck(FIXTURES / "norne_vfp_deck.DATA")
    fig = network_graph_figure(deck)
    assert len(fig.data) == 3  # 2 branch lines + 1 node trace
    assert "FIELD" in str(fig.data[-1].text)


# ------------------------------------------------------------------ M6 derive


def test_dp_curve_matches_hand_value():
    t = parse_vfp_file(TABLE_A)
    flo, dp = dp_curve(t, thp_index=0)
    # first THP=100, WCT=0, GOR=100, ALQ=0: BHP = 100+50+0.5*FLO+5 -> dp = 55+0.5*FLO
    np.testing.assert_allclose(dp, 55.0 + 0.5 * flo)


def test_gradient_constant_for_linear_table():
    t = parse_vfp_file(TABLE_A)
    mid, grad = gradient(t, thp_index=0)
    np.testing.assert_allclose(grad, 0.5)


def test_fit_recovers_quadratic_exponent(tmp_path):
    # table with pure friction dP = 20 + 0.01*Q^2
    flo = [100.0, 300.0, 600.0, 1000.0, 1500.0]
    p = tmp_path / "quad.inc"
    lines = [
        "VFPPROD\n 1 1500.0 'GAS' 'WGR' 'OGR' 'THP' ' ' 'METRIC' 'BHP' /",
        " " + " ".join(f"{q:g}" for q in flo) + " /",
        " 30 /", " 0.0 /", " 100 /", " 0.0 /",
        " 1 1 1 1 " + " ".join(f"{30 + 20 + 0.01 * q ** 2:.2f}" for q in flo) + " /",
        "/",
    ]
    p.write_text("\n".join(lines))
    t = parse_vfp_file(p)
    fit = fit_pressure_drop(t, thp_index=0)
    assert fit is not None
    assert fit["n"] == pytest.approx(2.0, abs=0.05)
    assert fit["a"] == pytest.approx(20.0, abs=0.5)


def test_turning_points_found():
    t = parse_vfp_file(FIXTURES / "norne_vfp_deck.DATA", table_number=9)
    tp = turning_points(t)
    assert tp, "pe2.VFP should have turning points (unstable limb)"
    assert all("flo" in p and "value" in p for p in tp)


def test_coverage_from_summary_with_lookup():
    """End-to-end: summary vectors through clamped lookup produce masks."""
    t = parse_vfp_file(TABLE_A)
    n = 50
    smry = SummaryFake({
        "WOPR:P1": np.linspace(5, 30, n),
        "WWPR:P1": np.zeros(n),
        "WTHP:P1": np.full(n, 150.0),
        "WWCT:P1": np.zeros(n),
        "WGOR:P1": np.full(n, 100.0),
    })
    rep = coverage_from_summary(t, "P1", smry)
    assert rep.fraction_clamped_low("FLO") > 0.0
    assert rep.fraction_clamped_high("FLO") > 0.0
