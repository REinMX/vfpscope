"""Plotly figure builders — pure functions of VfpTable; never render here.

Hard rule: no Streamlit/UI imports; figures are testable and reusable by the
CLI (PNG/HTML export) and the report builder.
"""

from __future__ import annotations

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


def coverage_overlay_figure(table, report) -> go.Figure:
    """Operating envelope vs table: clamped scatter + per-axis coverage bars.

    Top: run points (FLO vs WBHP, or FLO vs THP when WBHP is missing),
    coloured red when any axis was clamped; table's first THP curves drawn as
    reference. Bottom: per-axis stacked bars (in-range / clamped low / high).
    """
    flo = report.axis_values.get("FLO")
    if flo is None:
        raise ValueError("coverage report has no FLO data")
    bhp = report.bhp if report.bhp is not None else report.axis_values.get("THP")
    any_clamped = np.zeros(flo.size, dtype=bool)
    for a in report.axes_with_data:
        any_clamped |= report.clamped_low[a] | report.clamped_high[a]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("run envelope vs table (clamped points in red)",
                        "per-axis coverage (fraction of timesteps)"),
    )
    unit = table.axes["FLO"].unit
    fig.add_trace(
        go.Scatter(x=flo, y=bhp, mode="markers",
                   marker=dict(size=5, color=["#c0392b" if c else "#1f77b4" for c in any_clamped],
                               opacity=0.7),
                   name="run (clamped=red)",
                   hovertemplate="FLO: %{x:.6g}<br>BHP: %{y:.6g}<extra></extra>"),
        row=1, col=1,
    )
    # reference: first few table curves (THP family at fixed WFR/GFR/ALQ=0)
    try:
        ref = lift_curve_figure(table, x_axis="FLO", family="THP",
                                fixed={"WFR": 0, "GFR": 0, "ALQ": 0})
        for tr in ref.data:
            tr.showlegend = False
            tr.marker = dict(size=3)
            fig.add_trace(tr, row=1, col=1)
    except ValueError:
        pass
    axes = [a for a in report.axes_with_data]
    low = [report.fraction_clamped_low(a) for a in axes]
    high = [report.fraction_clamped_high(a) for a in axes]
    ok = [1.0 - lo - hi for lo, hi in zip(low, high, strict=True)]
    fig.add_trace(go.Bar(x=axes, y=ok, name="in range", marker_color="#1e8449"), row=2, col=1)
    fig.add_trace(go.Bar(x=axes, y=low, name="clamped low", marker_color="#b9770e"), row=2, col=1)
    fig.add_trace(go.Bar(x=axes, y=high, name="clamped high", marker_color="#c0392b"), row=2, col=1)
    fig.update_layout(
        barmode="stack",
        title=f"{table.label} — well {report.well} coverage ({report.n_timesteps} timesteps)",
        template="plotly_white",
        xaxis2_title=f"rate [{unit}]" if unit else "rate",
        yaxis2_title="fraction",
        hovermode="closest",
    )
    return fig


def compare_figure(tables: list, x_axis: str = "FLO", family: str = "THP",
                   fixed: dict[str, int] | None = None) -> go.Figure:
    """Overlay two+ tables on the same axes; unit-system guard per spec 3.5.

    Refuses (ValueError) when tables disagree on unit system or on the
    FLO/WFR/GFR/ALQ axis *types* — no silent conversion, no apples/oranges.
    """
    if len(tables) < 2:
        raise ValueError("compare needs at least two tables")
    t0 = tables[0]
    systems = {t.unit_system for t in tables}
    if len(systems) > 1:
        raise ValueError(
            "cannot overlay tables with differing unit systems "
            f"({systems}); convert explicitly first"
        )
    for axis in ("FLO", "WFR", "GFR", "ALQ"):
        kinds = {t.axes[axis].kind for t in tables}
        if len(kinds) > 1:
            raise ValueError(
                f"cannot overlay tables with differing {axis} types {kinds}"
            )
    fig = go.Figure()
    unit = t0.axes[family].unit
    for k, t in enumerate(tables):
        arr, xvals, famvals = _slice_and_reorder(t, x_axis, family, fixed or {})
        label = f"table {t.number}" + (f" ({t.label.split('(')[-1][:-1]})" if False else "")
        for fi in range(arr.shape[0]):
            fig.add_trace(
                go.Scatter(x=xvals, y=arr[fi], mode="lines+markers",
                           name=f"{label} THP={_fmt(famvals[fi], unit)}",
                           line=dict(color=_COLOR_SEQ[(k * 7 + fi) % len(_COLOR_SEQ)]),
                           marker=dict(size=4),
                           hovertemplate=f"{label}: %{{x:.6g}}, %{{y:.6g}}<extra></extra>"),
            )
    fig.update_layout(
        title=" vs ".join(f"table {t.number}" for t in tables),
        xaxis_title=_axis_label(t0, x_axis),
        yaxis_title=_y_label(t0),
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def compare_difference_figure(table_a, table_b, x_axis: str = "FLO") -> go.Figure:
    """Difference panel: table_A minus table_B interpolated onto A's grid."""
    a = table_a.data
    b = table_b.data
    if a.shape != b.shape:
        raise ValueError(
            "difference panel requires equal axis grids; "
            f"shapes {a.shape} vs {b.shape}"
        )
    fig = go.Figure()
    # one trace per THP slice at WFR=GFR=ALQ=0 (first slice), as a compact summary
    for ti in range(min(a.shape[0], 8)):
        fig.add_trace(
            go.Scatter(x=table_a.axes[x_axis].values, y=(a[ti, 0, 0, 0, :] - b[ti, 0, 0, 0, :]),
                       mode="lines+markers",
                       name=f"THP={table_a.axes['THP'].values[ti]:g}",
                       hovertemplate="%{x:.6g}, d(BHP)=%{y:.6g}<extra></extra>"),
        )
    fig.update_layout(
        title=f"table {table_a.number} − table {table_b.number} (BHP difference)",
        xaxis_title=_axis_label(table_a, x_axis),
        yaxis_title="difference",
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def network_graph_figure(deck) -> go.Figure:
    """Static directed graph of BRANPROP/NODEPROP; fixed-pressure nodes marked."""
    nodes = {name for name, _ in deck.nodes}
    for dt, ut, _ in deck.branches:
        nodes.add(dt)
        nodes.add(ut)
    nodes = sorted(nodes)
    fixed = {name for name, p in deck.nodes if p is not None}
    pos = {n: i for i, n in enumerate(nodes)}
    fig = go.Figure()
    for dt, ut, vfp in deck.branches:
        x0 = pos[dt]
        x1 = pos[ut]
        fig.add_trace(
            go.Scatter(x=[x0, x1], y=[0, 0], mode="lines+markers",
                       line=dict(color="#7f7f7f", width=2),
                       marker=dict(size=1),
                       text=[f"{dt}->{ut} (VFP {vfp})", ""],
                       hoverinfo="text",
                       name=f"{dt}->{ut}"),
        )
    fig.add_trace(
        go.Scatter(x=[pos[n] for n in nodes], y=[0] * len(nodes), mode="markers+text",
                   marker=dict(size=[26 if n in fixed else 16 for n in nodes],
                               color=["#c0392b" if n in fixed else "#1f77b4" for n in nodes]),
                   text=nodes, textposition="bottom center",
                   hovertemplate="%{text}<extra></extra>"),
    )
    fig.update_layout(
        title="network topology (red = fixed pressure)",
        template="plotly_white",
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=320,
    )
    return fig


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
