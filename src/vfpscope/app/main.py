"""Streamlit GUI for VFPScope.

Run via:  vfpscope serve deck.DATA
or:       streamlit run src/vfpscope/app/main.py -- --deck deck.DATA
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

from vfpscope.core.model import SEVERITY_ORDER
from vfpscope.core.parse.native import VfpParseError, load_deck

PRESETS = {
    "BHP vs FLO @ THP (well)": ("FLO", "THP"),
    "BHP vs WFR @ FLO": ("WFR", "FLO"),
    "THP vs FLO @ WFR (network)": ("FLO", "WFR"),
    "BHP vs ALQ @ FLO (gas lift)": ("ALQ", "FLO"),
}


@st.cache_resource(show_spinner="Parsing deck...")
def parse_deck(path: str):
    return load_deck(path)


def _deck_path_from_args() -> str | None:
    env = os.environ.get("VFPSCOPE_DECK")
    if env:
        return env
    args = sys.argv[1:]
    if "--deck" in args:
        i = args.index("--deck")
        if i + 1 < len(args):
            return args[i + 1]
    return None


def _table_options(deck) -> list[str]:
    out = []
    for no in deck.table_order:
        t = deck.tables[no]
        n_err = sum(1 for f in deck.findings if f.table_number == no and f.severity == "ERROR")
        n_warn = sum(1 for f in deck.findings if f.table_number == no and f.severity == "WARNING")
        badge = ""
        if n_err:
            badge = " [red]✗[/]"
        elif n_warn:
            badge = " [yellow]![/]"
        consumers = []
        if t.consumers.wells:
            consumers.append(f"{len(t.consumers.wells)} wells")
        if t.consumers.branches:
            consumers.append(f"{len(t.consumers.branches)} branches")
        cons = f" ({', '.join(consumers)})" if consumers else ""
        out.append(f"#{no} {t.kind} {t.role}{badge}{cons}")
    return out


def main() -> None:
    st.set_page_config(page_title="VFPScope", layout="wide", page_icon="📈")
    st.title("VFPScope — VFP table visualizer")

    arg_deck = _deck_path_from_args()
    deck_path = st.sidebar.text_input("Deck / include path", value=arg_deck or "")
    if not deck_path:
        st.info("Enter a deck path (or run `vfpscope serve <deck>`).")
        return
    if not os.path.exists(deck_path):
        st.error(f"file not found: {deck_path}")
        return
    try:
        deck = parse_deck(str(Path(deck_path).resolve()))
    except VfpParseError as e:
        st.error(str(e))
        return

    st.sidebar.caption(f"{len(deck.tables)} table(s), "
                       f"{sum(1 for f in deck.findings if f.severity == 'ERROR')} error(s), "
                       f"{sum(1 for f in deck.findings if f.severity == 'WARNING')} warning(s)")

    options = _table_options(deck)
    sel = st.sidebar.selectbox("Table", options, index=0)
    table_no = int(sel.split(" ")[0].lstrip("#"))
    table = deck.tables[table_no]

    tab_curves, tab_heatmap, tab_qc, tab_compare, tab_coverage = st.tabs(
        ["Curves", "Heatmap", "QC", "Compare", "Coverage"]
    )

    with tab_curves:
        _view_curves(deck, table)
    with tab_heatmap:
        _view_heatmap(deck, table)
    with tab_qc:
        _view_qc(deck, table)
    with tab_compare:
        _view_compare(deck, table)
    with tab_coverage:
        _view_coverage(deck, table)


def _fixed_sliders(table, exclude: set[str], key_prefix: str = "s"):
    """Sliders for every non-degenerate axis not used as x/family (M3)."""
    fixed: dict[str, int] = {}
    for a in ("THP", "WFR", "GFR", "ALQ", "FLO"):
        vals = table.axes[a].values
        if vals.size <= 1 or a in exclude:
            continue  # degenerate axes hidden, used axes excluded
        i = st.sidebar.slider(
            f"{a} ({table.axes[a].kind})",
            min_value=0,
            max_value=vals.size - 1,
            value=0,
            key=f"{key_prefix}_{table.number}_{a}",
        )
        fixed[a] = int(i)
    return fixed


def _view_curves(deck, table) -> None:
    from vfpscope.viz.figures import branch_figure, lift_curve_figure

    axes = [a for a in ("FLO", "THP", "WFR", "GFR", "ALQ") if table.axis_lengths[a] > 1]
    preset = st.sidebar.selectbox("Preset", list(PRESETS), index=0)
    px, pfamily = PRESETS[preset]
    x_axis = st.sidebar.selectbox("x axis", axes, index=axes.index(px) if px in axes else 0)
    family = st.sidebar.selectbox(
        "curve family", [a for a in axes if a != x_axis],
        index=0,
    )
    fixed = _fixed_sliders(table, exclude={x_axis, family}, key_prefix="cv")
    show_hints = st.sidebar.checkbox("Show QC plot hints", value=False)

    if table.role == "BRANCH":
        fig = branch_figure(table, x_axis=x_axis, family=family, fixed=fixed)
    else:
        fig = lift_curve_figure(table, x_axis=x_axis, family=family, fixed=fixed)
    if show_hints:
        from vfpscope.viz.figures import apply_plot_hint

        for f in deck.findings:
            if f.table_number == table.number:
                apply_plot_hint(fig, f.plot_hint)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Table info"):
        st.json(
            {
                "number": table.number,
                "kind": table.kind,
                "role": table.role,
                "datum_depth": table.datum_depth,
                "unit_system": table.unit_system,
                "tabulated": table.tabulated,
                "axis_lengths": table.axis_lengths,
                "axis_kinds": {a: table.axes[a].kind for a in table.axes},
                "consumers": {
                    "wells": list(table.consumers.wells),
                    "branches": [list(b) for b in table.consumers.branches],
                },
                "source": str(table.source),
            }
        )


def _view_heatmap(deck, table) -> None:
    from vfpscope.viz.figures import heatmap_figure

    axes = [a for a in ("FLO", "THP", "WFR", "GFR", "ALQ") if table.axis_lengths[a] > 1]
    if len(axes) < 2:
        st.info("Need at least two non-degenerate axes for a heatmap.")
        return
    x_axis = st.sidebar.selectbox("heatmap x", axes, index=0)
    y_axis = st.sidebar.selectbox(
        "heatmap y", [a for a in axes if a != x_axis], index=0
    )
    fixed = _fixed_sliders(table, exclude={x_axis, y_axis}, key_prefix="hm")
    as_contour = st.sidebar.checkbox("Contour (vs 3-D surface)", value=True)
    st.plotly_chart(
        heatmap_figure(table, x_axis=x_axis, y_axis=y_axis, fixed=fixed, as_contour=as_contour),
        use_container_width=True,
    )


def _view_qc(deck, table) -> None:
    findings = [f for f in deck.findings if f.table_number == table.number]
    all_findings = [f for f in deck.findings if f.table_number == 0]  # deck-level
    if not findings and not all_findings:
        st.success("No QC findings for this table.")
        return
    color = {"ERROR": "🔴", "WARNING": "🟡", "INFO": "⚪"}
    for f in sorted(findings + all_findings, key=lambda x: -SEVERITY_ORDER[x.severity]):
        locus = f"  \n`{f.locus}`" if f.locus else ""
        st.markdown(f"**{color[f.severity]} {f.check_id}** ({f.severity}) — {f.message}{locus}")
    st.caption("Run `vfpscope qc <deck> --fail-on warning` in a terminal for exit-code gating.")


def _view_compare(deck, table) -> None:
    st.info("Compare mode lands with the M6 milestone: pick a second deck/table to overlay.")
    other = st.sidebar.text_input("Second deck path (compare)")
    if other and os.path.exists(other):
        try:
            other_deck = parse_deck(str(Path(other).resolve()))
            st.write(f"Second deck: {len(other_deck.tables)} table(s).")
            st.warning("Overlay implementation pending (M6).")
        except VfpParseError as e:
            st.error(str(e))


def _view_coverage(deck, table) -> None:
    st.info("Coverage overlay against simulation output lands with the M5 milestone.")
    smry = st.sidebar.text_input("UNSMRY path (coverage)")
    if smry and os.path.exists(smry):
        st.warning("Coverage module pending (M5).")


if __name__ == "__main__":
    main()
