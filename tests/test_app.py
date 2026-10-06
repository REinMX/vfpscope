"""Streamlit AppTest smoke tests for the GUI shell."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FIXTURES = Path(__file__).parent / "fixtures"
APP = Path(__file__).resolve().parent.parent / "src" / "vfpscope" / "app" / "main.py"

pytestmark = pytest.mark.streamlit


def _table_select(at):
    return next(
        box
        for box in at.selectbox
        if any("#1 VFPPROD WELL" in str(option) for option in box.options)
    )


def test_app_loads_deck_and_renders_table_selector():
    os.environ["VFPSCOPE_DECK"] = str(FIXTURES / "norne_vfp_deck.DATA")
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    tables_sel = _table_select(at)
    assert any("#1 VFPPROD WELL" in o for o in tables_sel.options)
    assert len(at.slider) >= 2


def test_app_switching_table_reruns_cleanly():
    os.environ["VFPSCOPE_DECK"] = str(FIXTURES / "norne_vfp_deck.DATA")
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception
    tables_sel = _table_select(at)
    idx = next(i for i, o in enumerate(tables_sel.options) if "#6 " in o)
    tables_sel.set_value(tables_sel.options[idx]).run()
    assert not at.exception, at.exception


def test_app_missing_file_shows_error():
    os.environ["VFPSCOPE_DECK"] = "/nonexistent/deck.DATA"
    at = AppTest.from_file(str(APP), default_timeout=15)
    at.run()
    assert not at.exception
    assert any("file not found" in e.value for e in at.error)


def _page_text(at) -> str:
    chunks = []
    for name in ("markdown", "caption", "info", "success", "warning", "error"):
        for element in getattr(at, name):
            chunks.append(element.value)
    for element in at.number_input:
        chunks.append(element.label)
    return "\n".join(chunks)


def test_app_shows_petrel_run_gate_and_well_assignment():
    os.environ["VFPSCOPE_DECK"] = str(FIXTURES / "norne_vfp_deck.DATA")
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    text = _page_text(at)
    assert "Do not run" in text
    assert "B1" in text
    assert "Water cut" in text


def test_app_compare_empty_deck_does_not_crash(tmp_path):
    empty = tmp_path / "empty.DATA"
    empty.write_text("METRIC\n")
    os.environ["VFPSCOPE_DECK"] = str(FIXTURES / "norne_vfp_deck.DATA")
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    second = next(item for item in at.text_input if "Second deck" in item.label)
    second.set_value(str(empty)).run()
    assert not at.exception, at.exception
    assert "no vfp table" in _page_text(at).lower()


def test_app_lookup_outside_rate_says_not_an_eclipse_run():
    os.environ["VFPSCOPE_DECK"] = str(FIXTURES / "norne_vfp_deck.DATA")
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    flo = next(item for item in at.number_input if item.label.startswith("Liquid rate"))
    flo.set_value(flo.value + 1.0e6).run()
    assert not at.exception, at.exception
    text = _page_text(at)
    assert "extrapolat" in text.lower()
    assert "not an Eclipse run" in text
