"""Acceptance tests for the copy-paste standalone VFPScope application."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

FIXTURES = Path(__file__).parent / "fixtures"
STANDALONE = Path(__file__).resolve().parent.parent / "standalone" / "vfpscope_standalone.py"


def _load_standalone():
    spec = importlib.util.spec_from_file_location("vfpscope_standalone", STANDALONE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_standalone_is_one_self_contained_python_file():
    assert STANDALONE.is_file()
    source = STANDALONE.read_text()
    assert "from vfpscope" not in source
    assert "import vfpscope" not in source


def test_standalone_imports_only_declared_runtime_dependencies():
    tree = ast.parse(STANDALONE.read_text())
    import_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            import_roots.add(node.module.split(".")[0])
    stdlib = {
        "__future__",
        "collections",
        "dataclasses",
        "html",
        "itertools",
        "os",
        "pathlib",
        "re",
        "sys",
        "typing",
    }
    assert import_roots - stdlib == {
        "numpy",
        "plotly",
        "pydantic",
        "resdata",
        "streamlit",
    }


def test_standalone_parser_and_qc_match_expected_fixture_behavior():
    standalone = _load_standalone()
    deck = standalone.load_deck(FIXTURES / "norne_vfp_deck.DATA")
    assert deck.table_order == [1, 2, 3, 4, 5, 6, 8, 9, 12, 17]
    findings = standalone.run_qc(deck)
    assert {f.table_number for f in findings if f.check_id == "ROLE_CONFLICT"} == {8, 9}
    assert any(f.check_id == "UNSTABLE_BRANCH" and f.table_number == 9 for f in findings)


def test_standalone_builds_interactive_lift_curve():
    standalone = _load_standalone()
    table = standalone.parse_vfp_file(FIXTURES / "synthetic_2x3x2x2x2.inc")
    figure = standalone.lift_curve_figure(table)
    assert len(figure.data) == 3
    np.testing.assert_allclose(figure.data[0].y, table.data[0, 0, 0, 0, :])


def test_standalone_builds_summary_coverage_report_and_overlay():
    standalone = _load_standalone()
    table = standalone.parse_vfp_file(FIXTURES / "synthetic_2x3x2x2x2.inc")

    class SummaryFake:
        vectors = {
            "WOPR:PROD": np.array([5.0, 10.0, 20.0]),
            "WWPR:PROD": np.array([5.0, 10.0, 10.0]),
            "WTHP:PROD": np.array([100.0, 200.0, 300.0]),
            "WWCT:PROD": np.array([0.0, 0.5, 1.0]),
            "WGOR:PROD": np.array([100.0, 150.0, 200.0]),
            "WBHP:PROD": np.array([160.0, 220.0, 310.0]),
        }

        def numpy_vector(self, key: str) -> np.ndarray:
            return self.vectors[key]

    report = standalone.coverage_from_summary(table, "PROD", SummaryFake())
    assert report.n_timesteps == 3
    assert report.fraction_clamped_high("FLO") > 0.0
    figure = standalone.coverage_overlay_figure(table, report)
    assert len(figure.data) >= 4


@pytest.mark.streamlit
def test_standalone_streamlit_app_loads_real_deck():
    os.environ["VFPSCOPE_DECK"] = str(FIXTURES / "norne_vfp_deck.DATA")
    app = AppTest.from_file(str(STANDALONE), default_timeout=15)
    app.run()
    assert not app.exception, app.exception
    assert "Coverage" in [tab.label for tab in app.tabs]
    assert any("#1 VFPPROD WELL" in option for option in app.selectbox[0].options)
    assert len(app.slider) >= 2
