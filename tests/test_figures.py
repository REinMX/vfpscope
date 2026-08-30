"""Figure builder tests: lift curves, branch view, pivot, heatmap, report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfpscope.core.parse.native import load_deck, parse_vfp_file
from vfpscope.viz.figures import (
    branch_figure,
    build_report_html,
    heatmap_figure,
    lift_curve_figure,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def table():
    return parse_vfp_file(FIXTURES / "synthetic_2x3x2x2x2.inc")


def test_lift_curve_trace_per_thp(table):
    fig = lift_curve_figure(table)
    assert len(fig.data) == 3  # NTHP
    assert list(fig.data[0].x) == [10.0, 20.0]
    np.testing.assert_allclose(fig.data[0].y, table.data[0, 0, 0, 0, :])
    np.testing.assert_allclose(fig.data[2].y, table.data[2, 0, 0, 0, :])


def test_lift_curve_fixed_axes_shift_curves(table):
    fig = lift_curve_figure(table, fixed={"WFR": 1, "GFR": 1, "ALQ": 0})
    np.testing.assert_allclose(fig.data[0].y, table.data[0, 1, 1, 0, :])


def test_pivot_on_gfr_axis(table):
    fig = lift_curve_figure(table, x_axis="GFR", family="THP", fixed={"FLO": 1})
    assert len(fig.data) == 3
    assert list(fig.data[0].x) == [100.0, 200.0]
    np.testing.assert_allclose(fig.data[0].y, table.data[0, 0, :, 0, 1])


def test_pivot_on_alq_axis(table):
    fig = lift_curve_figure(table, x_axis="ALQ", family="FLO", fixed={"THP": 0, "WFR": 0, "GFR": 0})
    assert len(fig.data) == 2  # NFLO
    assert list(fig.data[0].x) == [0.0, 50000.0]
    np.testing.assert_allclose(fig.data[0].y, table.data[0, 0, 0, :, 0])


def test_degenerate_axes_are_hidden_for_vfpinj():
    inj = parse_vfp_file(FIXTURES / "synthetic_vfpinj_3x3.inc")
    fig = lift_curve_figure(inj, x_axis="FLO", family="THP")
    assert len(fig.data) == 3
    with pytest.raises(ValueError):
        lift_curve_figure(inj, x_axis="WFR", family="THP")


def test_invalid_pivot_raises(table):
    with pytest.raises(ValueError):
        lift_curve_figure(table, x_axis="FLO", family="FLO")


def test_branch_figure_has_delta_p_panel():
    deck = load_deck(FIXTURES / "norne_vfp_deck.DATA")
    t8 = deck.tables[8]  # branch-ish table (pd2.VFP, datum 400)
    fig = branch_figure(t8)
    # 2 panels: inlet pressure (NTHP traces) + delta-P (NTHP traces)
    n = t8.axes["THP"].values.size
    assert len(fig.data) == 2 * n
    thp0 = t8.axes["THP"].values[0]
    np.testing.assert_allclose(fig.data[n].y, fig.data[0].y - thp0)


def test_heatmap_contour_shape(table):
    fig = heatmap_figure(table, x_axis="FLO", y_axis="THP", fixed={"WFR": 0, "GFR": 0, "ALQ": 0})
    z = np.asarray(fig.data[0].z)
    assert z.shape == (3, 2)  # (NTHP, NFLO)


def test_report_html_contains_tables_and_findings(tmp_path):
    deck = load_deck(FIXTURES / "norne_vfp_deck.DATA")
    out = tmp_path / "report.html"
    build_report_html(deck, out)
    html = out.read_text()
    assert "VFPScope report" in html
    assert "VFPPROD #1" in html
    assert "ROLE_CONFLICT" in html  # table 8 verdict listed
    assert html.count("VFPPROD #") >= 8


def test_png_export_via_kaleido(table, tmp_path):
    fig = lift_curve_figure(table)
    png = tmp_path / "lift.png"
    try:
        fig.write_image(str(png), width=800, height=500)
    except Exception as e:  # kaleido not available
        pytest.skip(f"kaleido unavailable: {e}")
    assert png.exists() and png.stat().st_size > 1000
