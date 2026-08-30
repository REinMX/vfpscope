"""Plotly figure builders — pure functions of VfpTable; never render here.

Hard rule: no Streamlit/UI imports; figures are testable and reusable by the
CLI (PNG/HTML export) and the report builder.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.model import AXIS_ORDER, SEVERITY_ORDER, VfpTable, resolve_axis_unit

_MAX_POINTS_PER_TRACE = 400

_COLOR_SEQ = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _display_axes(table: VfpTable) -> list[str]:
    """Axes with more than one value; degenerate (length-1) axes are hidden."""
    return [a for a in AXIS_ORDER if table.axis_lengths[a] > 1]


def _fmt(v: float, unit: str = "") -> str:
    s = f"{v:g}"
    return f"{s} {unit}".strip()


def _slice_and_reorder(
    table: VfpTable, x_axis: str, family: str, fixed: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a 2-D slab (family, x) with axis value vectors.

    ``fixed`` holds 0-based indices for every other axis (defaults applied by
    the caller for degenerate axes).
    """
    if x_axis == family:
        raise ValueError("x_axis and family must differ")
    slices = []
    for a in AXIS_ORDER:
        if a in (x_axis, family):
            slices.append(slice(None))
        else:
            slices.append(int(fixed.get(a, 0)))
    arr = table.data[tuple(slices)]  # dims: AXIS_ORDER members of {x,family}
    dims = [a for a in AXIS_ORDER if a in (x_axis, family)]
    arr = arr.transpose([dims.index(family), dims.index(x_axis)])
    return arr, table.axes[x_axis].values, table.axes[family].values


def _add_trace(fig, x, y, name, color, showlegend=True, dash=None):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=name,
            showlegend=showlegend,
            line=dict(color=color, dash=dash) if dash else dict(color=color),
            marker=dict(size=5),
            hovertemplate=f"<b>{name}</b><br>%{{x:.6g}}<br>%{{y:.6g}}<extra></extra>",
        )
    )


def _axis_label(table: VfpTable, axis: str) -> str:
    ax = table.axes[axis]
    unit = ax.unit or ""
    kind = ax.kind or axis
    if axis == "THP" and table.role == "BRANCH":
        base = "outlet pressure"
    elif axis == "FLO" and table.role == "BRANCH":
        base = "throughput"
    elif axis == "THP":
        base = "wellhead pressure"
    elif axis == "FLO":
        base = "rate"
    else:
        base = axis
    return f"{base} [{kind}, {unit}]" if unit else f"{base} [{kind}]"


def _y_label(table: VfpTable) -> str:
    unit = resolve_axis_unit(table.unit_system, "THP", "THP")
    qty = "inlet pressure" if table.role == "BRANCH" else table.tabulated
    return f"{qty} [{unit}]" if unit else qty


def lift_curve_figure(
    table: VfpTable,
    x_axis: str = "FLO",
    family: str = "THP",
    fixed: dict[str, int] | None = None,
    downsample: int = _MAX_POINTS_PER_TRACE,
    shade_coverage: bool = True,
) -> go.Figure:
    """One curve per ``family`` value; remaining axes fixed at given indices.

    x_axis and family can be any of the table's non-degenerate axes
    (generalised axis pivot, M3). Hover shows the exact tabulated values.
    """
    axes = _display_axes(table)
    if x_axis not in axes or family not in axes:
        raise ValueError(
            f"x_axis={x_axis}, family={family} must be non-degenerate axes of "
            f"the table (available: {axes})"
        )
    fixed = dict(fixed or {})
    for a in axes:
        fixed.setdefault(a, 0)
    arr, xvals, famvals = _slice_and_reorder(table, x_axis, family, fixed)
    nfam, nx = arr.shape

    fig = go.Figure()
    fig.update_layout(
        title=f"{table.label} — {family} curves @ " + ", ".join(
            f"{a}={table.axes[a].values[fixed[a]]:g}" for a in axes if a not in (x_axis, family)
        ),
        xaxis_title=_axis_label(table, x_axis),
        yaxis_title=_y_label(table),
        hovermode="closest",
        template="plotly_white",
        legend_title=family,
    )
    unit = table.axes[family].unit
    stride = max(1, int(np.ceil(nx / downsample))) if downsample else 1
    xs = xvals[::stride]
    for fi in range(nfam):
        ys = arr[fi][::stride]
        _add_trace(
            fig,
            xs,
            ys,
            f"{family}={_fmt(famvals[fi], unit)}",
            _COLOR_SEQ[fi % len(_COLOR_SEQ)],
        )
    if shade_coverage and table.axes[x_axis].values.size > 1:
        lo, hi = float(xvals[0]), float(xvals[-1])
        fig.add_vrect(x0=lo - (hi - lo), x1=lo, fillcolor="red", opacity=0.04, line_width=0)
        fig.add_vrect(x0=hi, x1=hi + (hi - lo), fillcolor="red", opacity=0.04, line_width=0)
    return fig


def branch_figure(
    table: VfpTable,
    x_axis: str = "FLO",
    family: str = "THP",
    fixed: dict[str, int] | None = None,
    downsample: int = _MAX_POINTS_PER_TRACE,
) -> go.Figure:
    """Branch (flowline) view: inlet pressure and delta-P vs throughput.

    Top panel: upstream/inlet node pressure per downstream/outlet pressure.
    Bottom panel: pressure drop across the branch (tabulated - outlet).
    """
    axes = _display_axes(table)
    fixed = dict(fixed or {})
    for a in axes:
        fixed.setdefault(a, 0)
    arr, xvals, famvals = _slice_and_reorder(table, x_axis, family, fixed)
    nfam, nx = arr.shape
    # outlet pressure broadcast to (nfam, nx)
    if family == "THP":
        outlet = np.broadcast_to(famvals[:, None], arr.shape)
    elif x_axis == "THP":
        outlet = np.broadcast_to(xvals[None, :], arr.shape)
    else:
        outlet = np.full(arr.shape, float(table.axes["THP"].values[fixed.get("THP", 0)]))
    dp = arr - outlet

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("inlet pressure vs throughput", "delta-P vs throughput"),
    )
    unit = table.axes[family].unit
    stride = max(1, int(np.ceil(nx / downsample))) if downsample else 1
    xs = xvals[::stride]
    colors = [_COLOR_SEQ[fi % len(_COLOR_SEQ)] for fi in range(nfam)]
    names = [f"{family}={_fmt(famvals[fi], unit)}" for fi in range(nfam)]
    # row 1: inlet pressure traces; row 2: delta-P traces (grouped)
    for fi in range(nfam):
        fig.add_trace(
            go.Scatter(x=xs, y=arr[fi][::stride], mode="lines+markers", name=names[fi],
                       line=dict(color=colors[fi]), marker=dict(size=5),
                       hovertemplate=f"<b>{names[fi]}</b><br>rate: %{{x:.6g}}<br>"
                                     f"inlet: %{{y:.6g}}<extra></extra>"),
            row=1, col=1,
        )
    for fi in range(nfam):
        fig.add_trace(
            go.Scatter(x=xs, y=dp[fi][::stride], mode="lines+markers", name=names[fi],
                       line=dict(color=colors[fi], dash="dot"), marker=dict(size=5),
                       showlegend=False,
                       hovertemplate=f"<b>{names[fi]}</b><br>rate: %{{x:.6g}}<br>"
                                     f"dP: %{{y:.6g}}<extra></extra>"),
            row=2, col=1,
        )
    unit_y = resolve_axis_unit(table.unit_system, "THP", "THP")
    fig.update_xaxes(title_text=_axis_label(table, x_axis), row=2, col=1)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text=f"inlet pressure [{unit_y}]" if unit_y else "inlet pressure", row=1, col=1)
    fig.update_yaxes(title_text=f"delta-P [{unit_y}]" if unit_y else "delta-P", row=2, col=1)
    fig.update_layout(
        title=table.label,
        template="plotly_white",
        hovermode="closest",
        legend_title=family,
    )
    return fig


def heatmap_figure(
    table: VfpTable,
    x_axis: str,
    y_axis: str,
    fixed: dict[str, int] | None = None,
    as_contour: bool = True,
) -> go.Figure:
    """Contour (default) or 3-D surface of the tabulated value over two axes."""
    axes = _display_axes(table)
    if x_axis not in axes or y_axis not in axes or x_axis == y_axis:
        raise ValueError(f"x_axis/y_axis must be distinct non-degenerate axes (available: {axes})")
    fixed = dict(fixed or {})
    for a in axes:
        fixed.setdefault(a, 0)
    slices = []
    for a in AXIS_ORDER:
        if a in (x_axis, y_axis):
            slices.append(slice(None))
        else:
            slices.append(int(fixed.get(a, 0)))
    z = table.data[tuple(slices)]  # dims in AXIS_ORDER order
    dims = [a for a in AXIS_ORDER if a in (x_axis, y_axis)]
    z = z.transpose([dims.index(y_axis), dims.index(x_axis)])  # (ny, nx)
    xvals = table.axes[x_axis].values
    yvals = table.axes[y_axis].values
    if as_contour:
        fig = go.Figure(
            go.Contour(
                z=z, x=xvals, y=yvals,
                colorscale="Viridis",
                colorbar=dict(title=_y_label(table)),
                hovertemplate=f"{x_axis}: %{{x:.6g}}<br>{y_axis}: %{{y:.6g}}<br>"
                              f"value: %{{z:.6g}}<extra></extra>",
            )
        )
    else:
        fig = go.Figure(
            go.Surface(
                z=z, x=xvals, y=yvals,
                colorscale="Viridis",
                colorbar=dict(title=_y_label(table)),
            )
        )
    fixed_desc = ", ".join(
        f"{a}={table.axes[a].values[fixed[a]]:g}" for a in axes if a not in (x_axis, y_axis)
    )
    fig.update_layout(
        title=f"{table.label} — {_y_label(table)} over {y_axis} vs {x_axis}"
        + (f" @ {fixed_desc}" if fixed_desc else ""),
        xaxis_title=_axis_label(table, x_axis),
        yaxis_title=_axis_label(table, y_axis),
        template="plotly_white",
    )
    return fig


def apply_plot_hint(fig: go.Figure, hint: dict | None) -> None:
    """Highlight the problem locus described by a Finding.plot_hint."""
    if not hint:
        return
    kind = hint.get("kind")
    if kind == "turning_point" and "flo" in hint and "value" in hint:
        fig.add_trace(
            go.Scatter(
                x=[hint["flo"]], y=[hint["value"]], mode="markers",
                marker=dict(size=14, color="red", symbol="x"),
                name="turning point", hovertemplate="turning point: %{x:.6g}<extra></extra>",
            )
        )
    elif kind == "crossing" and "x" in hint and "y" in hint:
        fig.add_trace(
            go.Scatter(
                x=hint["x"], y=hint["y"], mode="markers",
                marker=dict(size=10, color="red", symbol="circle-open"),
                name="crossing", hovertemplate="curve crossing<extra></extra>",
            )
        )
    elif kind == "clamped" and "x" in hint and "y" in hint:
        fig.add_trace(
            go.Scatter(
                x=hint["x"], y=hint["y"], mode="markers",
                marker=dict(size=6, color="red", opacity=0.6),
                name="clamped timesteps",
                hovertemplate="clamped: %{x:.6g}, %{y:.6g}<extra></extra>",
            )
        )


def build_report_html(deck, out) -> None:
    """Self-contained HTML: every table's figure + QC verdicts."""
    import html as _html

    sections: list[str] = []
    from ..core.qc.engine import run_qc

    all_findings = run_qc(deck)
    first = True
    for no in deck.table_order:
        t = deck.tables[no]
        fig = branch_figure(t) if t.role == "BRANCH" else lift_curve_figure(t)
        findings = [f for f in all_findings if f.table_number == no]
        worst = max((SEVERITY_ORDER[f.severity] for f in findings), default=0)
        badge = {2: "ERROR", 1: "WARNING", 0: "OK"}[worst]
        color = {"ERROR": "#c0392b", "WARNING": "#b9770e", "OK": "#1e8449"}[badge]
        fig_html = fig.to_html(
            full_html=False, include_plotlyjs="cdn" if first else False,
            config={"displayModeBar": True},
        )
        first = False
        meta = (
            f"<p>kind: {t.kind} · role: {t.role} · datum: {t.datum_depth:g} · "
            f"units: {t.unit_system} · axes: "
            + ", ".join(f"{a}={t.axes[a].kind}({t.axis_lengths[a]})" for a in AXIS_ORDER)
            + "</p>"
        )
        consumers = []
        if t.consumers.wells:
            consumers.append("wells: " + ", ".join(t.consumers.wells))
        if t.consumers.branches:
            consumers.append("branches: " + ", ".join(f"{a}->{b}" for a, b in t.consumers.branches))
        if consumers:
            meta += "<p>" + " · ".join(consumers) + "</p>"
        fl = "".join(
            f"<li style='color:{ {'ERROR':'#c0392b','WARNING':'#b9770e','INFO':'#7f8c8d'}[f.severity] }'>"
            f"<b>{f.check_id}</b> ({f.severity}): {_html.escape(f.message)}</li>"
            for f in findings
        )
        findings_html = f"<ul>{fl}</ul>" if fl else "<p>no findings</p>"
        sections.append(
            f"<section><h2 style='color:{color}'>[{badge}] {_html.escape(t.label)}</h2>"
            f"{meta}{fig_html}<h3>QC findings</h3>{findings_html}</section>"
        )
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>VFPScope report</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;"
        "padding:0 1em}h2{border-bottom:2px solid #ddd;padding-bottom:.2em}"
        "section{margin-bottom:3em}</style></head><body>"
        f"<h1>VFPScope report — {_html.escape(str(deck.path))}</h1>"
        + "".join(sections)
        + "</body></html>"
    )
    out.write_text(doc, encoding="utf-8")
