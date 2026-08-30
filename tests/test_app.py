"""Streamlit AppTest smoke tests for the GUI shell."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FIXTURES = Path(__file__).parent / "fixtures"
APP = Path(__file__).resolve().parent.parent / "src" / "vfpscope" / "app" / "main.py"

pytestmark = pytest.mark.streamlit


def test_app_loads_deck_and_renders_table_selector():
    os.environ["VFPSCOPE_DECK"] = str(FIXTURES / "norne_vfp_deck.DATA")
    at = AppTest.from_file(str(APP), default_timeout=15)
    at.run()
    assert not at.exception, at.exception
    # table selector shows the first table with role/consumer info
    tables_sel = at.selectbox[0]
    assert any("#1 VFPPROD WELL" in o for o in tables_sel.options)
    # free-axis sliders are rendered (WFR and GFR non-degenerate for table 1)
    assert len(at.slider) >= 2


def test_app_switching_table_reruns_cleanly():
    os.environ["VFPSCOPE_DECK"] = str(FIXTURES / "norne_vfp_deck.DATA")
    at = AppTest.from_file(str(APP), default_timeout=15)
    at.run()
    assert not at.exception
    tables_sel = at.selectbox[0]
    # pick the dead table (#6, UNKNOWN role)
    idx = next(i for i, o in enumerate(tables_sel.options) if "#6 " in o)
    tables_sel.set_value(tables_sel.options[idx])
    at.run()
    assert not at.exception, at.exception


def test_app_missing_file_shows_error():
    os.environ["VFPSCOPE_DECK"] = "/nonexistent/deck.DATA"
    at = AppTest.from_file(str(APP), default_timeout=15)
    at.run()
    assert not at.exception
    assert any("file not found" in e.value for e in at.error)
