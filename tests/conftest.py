"""Shared test fixtures/helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# opm-tests corpus archive (optional, big) — used by real-corpus tests
OPM_TESTS_ZIP = Path("/home/javier/Downloads/opm-tests-master.zip")

HAS_RES2DF = False
try:
    import res2df  # noqa: F401

    HAS_RES2DF = True
except ImportError:
    pass


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def corpus_zip():
    if not OPM_TESTS_ZIP.exists():
        pytest.skip("opm-tests corpus zip not present")
    import zipfile

    return zipfile.ZipFile(OPM_TESTS_ZIP)
