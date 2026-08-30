"""Cross-check the native parser against res2df.vfp (the reference parser).

Skipped when res2df/opm are not installed (the extra is optional at runtime
but required in CI). Assert identical axis vectors and identical hypercube
arrays on synthetic fixtures and on the real corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vfpscope.core.parse.native import load_deck

pytestmark = pytest.mark.skipif(
    pytest.importorskip("res2df") is None, reason="res2df not installed"
)

FIXTURES = Path(__file__).parent / "fixtures"


def _res2df_basic(content: str, keyword: str):
    from res2df import vfp

    return vfp.basic_data(content, keyword)


def assert_identical(deck, content: str, keyword: str) -> None:
    ref = _res2df_basic(content, keyword)
    ref_tables = ref if isinstance(ref, list) else [ref]
    assert len(ref_tables) == len(deck.tables), (
        f"table count mismatch: native {len(deck.tables)} vs res2df {len(ref_tables)}"
    )
    for r in ref_tables:
        if not r:  # empty dict (filtered)
            continue
        no = int(r["TABLE_NUMBER"])
        t = deck.tables[no]
        assert r["RATE_TYPE"].value == t.axes["FLO"].kind
        assert r["UNIT_TYPE"].value == t.unit_system
        np.testing.assert_allclose(r["FLOW_VALUES"], t.axes["FLO"].values)
        np.testing.assert_allclose(r["THP_VALUES"], t.axes["THP"].values)
        if keyword == "VFPPROD":
            assert r["WFR_TYPE"].value == t.axes["WFR"].kind
            assert r["GFR_TYPE"].value == t.axes["GFR"].kind
            np.testing.assert_allclose(r["WFR_VALUES"], t.axes["WFR"].values)
            np.testing.assert_allclose(r["GFR_VALUES"], t.axes["GFR"].values)
            np.testing.assert_allclose(r["ALQ_VALUES"], t.axes["ALQ"].values)
            # map res2df deck-order rows onto the canonical hypercube
            thp = np.asarray(r["THP_INDICES"], dtype=int) - 1
            wfr = np.asarray(r["WFR_INDICES"], dtype=int) - 1
            gfr = np.asarray(r["GFR_INDICES"], dtype=int) - 1
            alq = np.asarray(r["ALQ_INDICES"], dtype=int) - 1
            rows = np.asarray(r["BHP_TABLE"])
            got = np.full_like(t.data, np.nan)
            got[thp, wfr, gfr, alq, :] = rows
            np.testing.assert_allclose(got, t.data, equal_nan=True)
        else:
            thp = np.asarray(r["THP_INDICES"], dtype=int) - 1
            rows = np.asarray(r["BHP_TABLE"])
            got = np.full_like(t.data, np.nan)
            got[thp, 0, 0, 0, :] = rows
            np.testing.assert_allclose(got, t.data, equal_nan=True)


def test_synthetic_vfpprod_matches_res2df(fixtures):
    content = (fixtures / "synthetic_2x3x2x2x2.inc").read_text()
    deck = load_deck(fixtures / "synthetic_2x3x2x2x2.inc")
    assert_identical(deck, content, "VFPPROD")


def test_synthetic_vfpinj_matches_res2df(fixtures):
    content = (fixtures / "synthetic_vfpinj_3x3.inc").read_text()
    deck = load_deck(fixtures / "synthetic_vfpinj_3x3.inc")
    assert_identical(deck, content, "VFPINJ")


def test_master_tables_match_res2df():
    p = Path("/home/javier/worktrees/oil-gas-priority-toy/spikes/004-opm-flow-master/vfp_tables_master.inc")
    content = p.read_text()
    deck = load_deck(p)
    assert_identical(deck, content, "VFPPROD")


def test_norne_vfpinj_matches_res2df(fixtures):
    p = fixtures / "norne_C1H.Ecl"
    content = p.read_text()
    deck = load_deck(p)
    assert_identical(deck, content, "VFPINJ")


def test_model5_well_vfp_full_corpus_matches_res2df(corpus_zip):
    """The big GRAT corpus file (21x5x10x9x8 tables): every table identical."""
    name = "opm-tests-master/model5/include/well_vfp.ecl"
    data = corpus_zip.read(name).decode("utf-8", "replace")
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".ecl", delete=False) as fh:
        fh.write(data.encode())
        tmp = fh.name
    try:
        deck = load_deck(tmp)
    finally:
        Path(tmp).unlink()
    assert len(deck.tables) == 1, f"expected 1 table, got {len(deck.tables)}"
    t = deck.tables[1]
    assert t.data.shape == (5, 10, 9, 8, 21)
    assert t.axes["FLO"].kind == "LIQ"
    assert t.axes["ALQ"].kind == "GRAT"
    assert_identical(deck, data, "VFPPROD")
