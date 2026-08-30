"""Coverage tests: synthetic summary + real SPE1CASE1 UNSMRY (if resdata)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfpscope.core.coverage import (
    CoverageUnavailable,
    coverage_from_summary,
    load_summary,
)
from vfpscope.core.parse.native import load_deck, parse_vfp_file
from vfpscope.core.qc.engine import QCContext, run_qc

FIXTURES = Path(__file__).parent / "fixtures"


class SummaryFake:
    def __init__(self, vectors: dict[str, np.ndarray]):
        self._v = {k.upper(): v for k, v in vectors.items()}

    def numpy_vector(self, key: str) -> np.ndarray:
        k = key.upper()
        if k not in self._v:
            raise KeyError(k)
        return self._v[k]


def _liq_table_text() -> str:
    lines = [
        "VFPPROD\n 1 1500.0 'LIQ' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'BHP' /",
        " 500 1000 2000 3500 5000 /",
        " 10 30 60 /",
        " 0.0 0.5 0.8 /",
        " 50 150 /",
        " 0.0 /",
    ]
    for ti, thp in enumerate([10.0, 30.0, 60.0], start=1):
        for wi, wct in enumerate([0.0, 0.5, 0.8], start=1):
            for gi, gor in enumerate([50.0, 150.0], start=1):
                vals = " ".join(
                    f"{thp + 100 + 0.02 * f + 20 * wct + 0.05 * gor:.2f}"
                    for f in [500.0, 1000.0, 2000.0, 3500.0, 5000.0]
                )
                lines.append(f" {ti} {wi} {gi} 1 {vals} /")
    return "\n".join(lines) + "\n/\n"


@pytest.fixture(scope="module")
def liq_table(tmp_path_factory):
    p = tmp_path_factory.mktemp("cov") / "coverage_liq.inc"
    p.write_text(_liq_table_text())
    return parse_vfp_file(p)


def _run_vectors(n=200):
    flo = np.linspace(400, 8000, n)  # exceeds table max 5000 -> high clamping
    thp = np.linspace(5, 55, n)      # below table min 10 -> low clamping early
    wct = np.linspace(0.0, 0.9, n)
    gor = np.linspace(50, 200, n)
    return SummaryFake({
        "WOPR:PROD": flo * 0.7,
        "WWPR:PROD": flo * 0.3,
        "WTHP:PROD": thp,
        "WWCT:PROD": wct,
        "WGOR:PROD": gor,
        "WBHP:PROD": 300 + 0.02 * flo,
    })


def test_coverage_flo_high_clamping(liq_table):
    smry = _run_vectors()
    rep = coverage_from_summary(liq_table, "PROD", smry)
    assert rep.n_timesteps == 200
    # FLO runs to 8000 vs table max 5000 -> 39.5% of timesteps high-clamped
    assert rep.fraction_clamped_high("FLO") == pytest.approx(0.3947, abs=1e-3)
    # THP starts at 5 vs table min 10
    assert rep.fraction_clamped_low("THP") > 0.0
    ratio = rep.run_max_over_table("FLO", liq_table)
    assert ratio is not None and ratio > 1.5
    assert rep.bhp is not None


def test_coverage_inside_table_no_clamping(liq_table):
    table = liq_table
    smry = SummaryFake({
        "WOPR:PROD": np.linspace(600, 4800, 50),
        "WWPR:PROD": np.linspace(100, 200, 50),
        "WTHP:PROD": np.linspace(12, 55, 50),
        "WWCT:PROD": np.linspace(0.1, 0.7, 50),
        "WGOR:PROD": np.linspace(60, 140, 50),
    })
    rep = coverage_from_summary(table, "PROD", smry)
    assert rep.fraction_clamped("FLO") == 0.0
    assert rep.fraction_clamped("THP") == 0.0


def test_coverage_missing_vectors_raises(liq_table):
    smry = SummaryFake({"WOPR:PROD": np.zeros(10)})
    with pytest.raises(CoverageUnavailable):
        coverage_from_summary(liq_table, "PROD", smry)


def test_coverage_checks_21_22_23(liq_table, tmp_path_factory):
    smry = _run_vectors()
    rep = coverage_from_summary(liq_table, "PROD", smry)
    p = tmp_path_factory.mktemp("cov2") / "coverage_liq.inc"
    p.write_text(_liq_table_text())
    deck = load_deck(p)
    ctx = QCContext(deck=deck, coverage={liq_table.number: [rep]})
    fs = run_qc(deck, context=ctx)
    ids = {f.check_id for f in fs}
    assert "CLAMP_FRACTION" in ids
    assert "RUN_MAX_EXCEEDS" in ids
    # FLO spends much of the run above the turning-point-free table -> no unstable
    # (table is monotone, so PERSISTENT_UNSTABLE must not fire)
    assert "PERSISTENT_UNSTABLE" not in ids
    clamp = next(f for f in fs if f.check_id == "CLAMP_FRACTION")
    assert clamp.severity == "WARNING"
    assert "PROD" in clamp.message


def test_coverage_persistent_unstable():
    # unstable table: BHP dips at 2000; run above 2000 most of the time
    p = FIXTURES / "coverage_unstable.inc"
    p.write_text(
        "VFPPROD\n 1 1500.0 'LIQ' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'BHP' /\n"
        " 500 1000 2000 3500 5000 /\n"
        " 30 /\n 0.0 /\n 100 /\n 0.0 /\n"
        " 1 1 1 1 160 140 130 155 170 /\n/\n"
    )
    table = parse_vfp_file(p)
    smry = SummaryFake({
        "WOPR:PROD": np.linspace(1500, 4500, 100),
        "WWPR:PROD": np.linspace(100, 300, 100),
        "WTHP:PROD": np.full(100, 30.0),
        "WWCT:PROD": np.zeros(100),
        "WGOR:PROD": np.full(100, 100.0),
    })
    rep = coverage_from_summary(table, "PROD", smry)
    deck = load_deck(p)
    ctx = QCContext(deck=deck, coverage={1: [rep]})
    fs = run_qc(deck, context=ctx)
    ids = {f.check_id for f in fs}
    assert "UNSTABLE_BRANCH" in ids
    assert "PERSISTENT_UNSTABLE" in ids


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("resdata") is None,
    reason="resdata not installed",
)
def test_real_spe1case1_summary():
    """Real-run evidence: SPE1CASE1 summary against a synthetic oil table."""
    smry = load_summary("/home/javier/spe1latest/SPE1CASE1.UNSMRY")
    assert smry is not None
    # table: OIL FLO, WCT, GOR, THP; FLO range below the run's WOPR max
    p = FIXTURES / "coverage_spe1.inc"
    p.write_text(
        "VFPPROD\n 1 1500.0 'OIL' 'WCT' 'GOR' 'THP' ' ' 'METRIC' 'BHP' /\n"
        " 100 200 300 400 500 /\n"
        " 10 30 60 /\n 0.0 0.5 /\n 50 150 300 /\n 0.0 /\n"
        + "".join(
            f" {ti} {wi} {gi} 1 "
            + " ".join(f"{thp + 150 + 0.1 * f + 10 * wct + 0.02 * gor:.2f}" for f in [100.0, 200.0, 300.0, 400.0, 500.0])
            + " /\n"
            for ti, thp in enumerate([10.0, 30.0, 60.0], start=1)
            for wi, wct in enumerate([0.0, 0.5], start=1)
            for gi, gor in enumerate([50.0, 150.0, 300.0], start=1)
        ) + "/\n"
    )
    table = parse_vfp_file(p)
    try:
        rep = coverage_from_summary(table, "PROD", smry)
    except CoverageUnavailable:
        pytest.skip("SPE1 summary lacks the well vectors")
    assert rep.n_timesteps > 10
    # SPE1 oil rate exceeds 500 sm3/d at some point -> some high FLO clamping
    assert rep.run_max_over_table("FLO", table) is not None
