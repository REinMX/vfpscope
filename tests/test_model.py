"""Model tests: canonical axis order, validation, units, consumers."""

from __future__ import annotations

import numpy as np
import pytest

from vfpscope.core.model import AXIS_ORDER, Consumers, SourceRef, VfpAxis, VfpTable


def _axes(**kwargs):
    defaults = {
        "FLO": ("LIQ", np.array([10.0, 20.0])),
        "THP": ("THP", np.array([100.0, 200.0])),
        "WFR": ("WCT", np.array([0.0, 0.5])),
        "GFR": ("GOR", np.array([100.0, 200.0])),
        "ALQ": ("GRAT", np.array([0.0])),
    }
    defaults.update(kwargs)
    return {
        name: VfpAxis(name=name, kind=kind, values=vals)  # type: ignore[arg-type]
        for name, (kind, vals) in defaults.items()
    }


def _table(**overrides):
    axes = overrides.pop("axes", _axes())
    shape = tuple(len(axes[a].values) for a in AXIS_ORDER if a in axes)
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    params = dict(
        number=1,
        kind="VFPPROD",
        role="UNKNOWN",
        datum_depth=1500.0,
        unit_system="METRIC",
        tabulated="BHP",
        axes=axes,
        data=data,
        source=SourceRef(path="x.inc", line=1),
    )
    params.update(overrides)
    return VfpTable(**params)


def test_canonical_axis_order():
    assert AXIS_ORDER == ("THP", "WFR", "GFR", "ALQ", "FLO")


def test_valid_table_constructs_and_data_shape():
    t = _table()
    assert t.data.shape == (2, 2, 2, 1, 2)


def test_axis_values_must_be_strictly_increasing():
    with pytest.raises(ValueError):
        _axes(FLO=("LIQ", np.array([10.0, 10.0])))
    with pytest.raises(ValueError):
        _axes(FLO=("LIQ", np.array([20.0, 10.0])))


def test_axis_values_must_be_1d():
    with pytest.raises(ValueError):
        _axes(FLO=("LIQ", np.array([[1.0], [2.0]])))


def test_data_shape_must_match_axes():
    with pytest.raises(ValueError):
        _table(data=np.zeros((2, 2, 2, 1, 3)))


def test_vfpinj_kind_allows_degenerate_axes():
    axes = _axes(WFR=("WCT", np.array([0.0])), GFR=("GOR", np.array([100.0])))
    t = _table(kind="VFPINJ", axes=axes)
    assert t.data.shape == (2, 1, 1, 1, 2)


def test_invalid_kind_rejected():
    with pytest.raises(ValueError):
        _table(kind="BOGUS")  # type: ignore[arg-type]


def test_missing_axis_rejected():
    axes = _axes()
    del axes["ALQ"]
    with pytest.raises(ValueError):
        _table(axes=axes)


def test_consumers_roundtrip():
    c = Consumers(wells=("P1", "P2"), branches=(("M5S", "PLAT-A"),))
    t = _table(consumers=c, role="WELL")
    assert t.consumers.wells == ("P1", "P2")
    assert t.role == "WELL"


def test_source_str():
    assert str(SourceRef(path="a/b.inc", line=42)) == "a/b.inc:42"


def test_axis_unit_resolution():
    ax = VfpAxis(name="FLO", kind="LIQ", values=np.array([1.0]), unit="sm3/day")
    assert ax.unit == "sm3/day"
