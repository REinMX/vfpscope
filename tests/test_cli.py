"""CLI tests (Typer CliRunner): list / qc / plot behaviours."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from vfpscope.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def test_list_on_norne_deck():
    r = runner.invoke(app, ["list", str(FIXTURES / "norne_vfp_deck.DATA")])
    assert r.exit_code == 0, r.output
    assert "10 table(s)" in r.output
    assert "VFPPROD" in r.output
    assert "VFPINJ" in r.output
    assert "WELL" in r.output
    assert "UNKNOWN" in r.output  # dead table 6
    assert "branch" in r.output  # table 8 has branch consumers


def test_list_on_missing_file_exits_2():
    r = runner.invoke(app, ["list", "/nonexistent/x.DATA"])
    assert r.exit_code == 2


def test_list_on_broken_table_still_lists_others():
    r = runner.invoke(app, ["list", str(FIXTURES / "broken_count_23of24.inc")])
    assert r.exit_code == 0
    assert "1 error(s)" in r.output


def test_qc_clean_deck_exits_zero():
    r = runner.invoke(app, ["qc", str(FIXTURES / "synthetic_2x3x2x2x2.inc")])
    assert r.exit_code == 0, r.output


def test_qc_broken_deck_fails_with_fail_on_error():
    r = runner.invoke(app, ["qc", str(FIXTURES / "broken_count_23of24.inc")])
    assert r.exit_code == 1
    assert "expected 24 data records, found 23" in r.output


def test_qc_invalid_fail_on_level_exits_2():
    r = runner.invoke(app, ["qc", str(FIXTURES / "synthetic_2x3x2x2x2.inc"), "--fail-on", "bogus"])
    assert r.exit_code == 2


def test_qc_norne_deck_has_expected_findings():
    # the real Norne flowline table 9 (pe2.VFP) is genuinely corrupt
    # (BHP dips as THP rises) -> ERROR findings -> exit 1
    r = runner.invoke(app, ["qc", str(FIXTURES / "norne_vfp_deck.DATA")])
    assert r.exit_code == 1
    assert "ROLE_CONFLICT" in r.output  # table 8/9 used by well + branch
    assert "THP_MONOTONIC" in r.output
    assert "DEAD_TABLE" in r.output


def test_plot_writes_html_by_default(tmp_path):
    r = runner.invoke(app, ["plot", str(FIXTURES / "synthetic_2x3x2x2x2.inc"), "--table", "1"])
    assert r.exit_code == 0, r.output
    out = FIXTURES / "synthetic_2x3x2x2x2.vfp_plot.html"
    assert out.exists()
    out.unlink()
