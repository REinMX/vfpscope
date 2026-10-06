"""Streamlit GUI for VFPScope.

Run via:  vfpscope serve deck.DATA
or:       streamlit run src/vfpscope/app/main.py -- --deck deck.DATA

The screen is a Petrel simulation check: which well uses which table, whether
the export is safe to run, and what bottom-hole pressure Eclipse will interpolate
— or extrapolate — at a planned operating point.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

from vfpscope.core.model import AXIS_ORDER
from vfpscope.core.parse.native import VfpParseError, load_deck
from vfpscope.core.petrel import (
    assignments_csv,
    axis_caption,
    evaluate_operating_point,
    finding_implication,
    quantity_caption,
    simulation_brief,
)
from vfpscope.core.qc.engine import run_qc

PRESETS = {
    "Bottom-hole pressure vs rate, by tubing-head pressure": ("FLO", "THP"),
    "Bottom-hole pressure vs water fraction, by rate": ("WFR", "FLO"),
    "Inlet pressure vs rate (network branch)": ("FLO", "WFR"),
    "Bottom-hole pressure vs gas-lift rate": ("ALQ", "FLO"),
}


@st.cache_resource(show_spinner="Reading Petrel export...")
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


def _table_label(deck, number: int, findings) -> str:
    table = deck.tables[number]
    n_err = sum(1 for f in findings if f.table_number == number and f.severity == "ERROR")
    n_warn = sum(1 for f in findings if f.table_number == number and f.severity == "WARNING")
    badge = ""
    if n_err:
        badge = f" {n_err} errors"
    elif n_warn:
        badge = f" {n_warn} warnings"
    consumers = []
    if table.consumers.wells:
        consumers.append(", ".join(table.consumers.wells))
    if table.consumers.branches:
        consumers.append(f"{len(table.consumers.branches)} branches")
    cons = f" {'; '.join(consumers)}" if consumers else ""
    return f"#{number} {table.kind} {table.role}{badge}{cons}"


def _chart(fig) -> None:
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.set_page_config(page_title="VFPScope", layout="wide", page_icon="📈")
    st.title("VFPScope")
    st.caption("Petrel simulation — check VFP tables before the Eclipse run.")

    arg_deck = _deck_path_from_args()
    deck_path = st.sidebar.text_input(
        "Petrel Eclipse export (.DATA or VFP include)",
        value=arg_deck or "",
    )
    if not deck_path:
        st.info(
            "Open the Eclipse deck exported from the Petrel simulation case.\n\n"
            "On this machine: `vfpscope serve CASE.DATA`\n\n"
            "A single VFP include also works. If the deck uses INCLUDE, open the "
            ".DATA file so those paths resolve next to it. VFPScope reads and "
            "judges tables. It does not build them."
        )
        return
    if not os.path.exists(deck_path):
        st.error(f"file not found: {deck_path}")
        return
    try:
        deck = parse_deck(str(Path(deck_path).resolve()))
    except VfpParseError as exc:
        st.error(str(exc))
        return

    brief = simulation_brief(deck)
    findings = run_qc(deck)
    if brief.gate == "DO_NOT_RUN":
        st.error(brief.headline)
    elif brief.gate == "REVIEW":
        st.warning(brief.headline)
    else:
        st.success(brief.headline)

    st.sidebar.caption(
        f"{brief.n_tables} table(s) · {brief.unit_system} · "
        f"{brief.n_errors} error(s) · {brief.n_warnings} warning(s)"
    )

    well_ids = ["(all wells)"] + [row.well for row in brief.wells]
    well_sel = st.sidebar.selectbox("Well", well_ids, key="petrel_well")
    selected_well = None if well_sel == "(all wells)" else well_sel
    numbers = _tables_for_well(deck, selected_well)
    if not numbers:
        if selected_well is None:
            st.error("No VFP tables in this export.")
            return
        missing = deck.well_vfp.get(selected_well)
        st.error(
            f"Well {selected_well} references VFP table {missing}, which is not in this export."
        )
        return
    labels = [_table_label(deck, number, findings) for number in numbers]
    selected = st.sidebar.selectbox("Table", labels, key=f"petrel_table_{well_sel}")
    table_no = int(selected.split()[0].lstrip("#"))
    table = deck.tables[table_no]

    tab_case, tab_lookup, tab_curves, tab_qc, tab_more = st.tabs(
        ["Case", "Lookup", "Curves", "QC", "More"]
    )
    with tab_case:
        _view_case(deck, brief)
    with tab_lookup:
        _view_lookup(table)
    with tab_curves:
        _view_curves(findings, table)
    with tab_qc:
        _view_qc(findings, table)
    with tab_more:
        _view_more(deck, table)


def _tables_for_well(deck, well: str | None) -> list[int]:
    if well is None:
        return list(deck.table_order)
    number = deck.well_vfp.get(well)
    if number in deck.tables:
        return [number]
    return []


def _view_case(deck, brief) -> None:
    st.markdown(
        f"Unit system **{brief.unit_system}**. "
        "Well rows are the VFP table numbers written in `WCONPROD` / `WCONINJE`. "
        "This does not read the Petrel control mode. THP control needs a table; "
        "BHP control does not use it to solve the rate."
    )
    if not brief.wells:
        st.info("No well references this export. Open the SCHEDULE include from the Petrel case, or the .DATA file that includes it.")
    else:
        st.markdown("**Wells assigned in the export**")
        for row in brief.wells:
            phase = row.phase or "phase unset"
            missing = "" if row.table_present else " — table missing"
            lift = " · gas lift" if row.gas_lift else ""
            datum = f", datum {row.datum:g}" if row.datum is not None else ""
            st.markdown(
                f"- **{row.well}** ({phase}{datum}): VFP {row.table_number}{missing}{lift}"
            )
        st.download_button(
            "Download well assignments (CSV)",
            data=assignments_csv(brief),
            file_name="vfp_well_assignments.csv",
            mime="text/csv",
            key="assign_csv",
        )
    if brief.unassigned_tables:
        listed = ", ".join(f"#{number}" for number in brief.unassigned_tables)
        st.markdown(f"Unassigned tables (no well or branch uses them): {listed}")
    if deck.branches:
        st.caption(
            f"{len(deck.branches)} network branch(es). Open More → Network to see which "
            "flowline uses which table."
        )


def _view_lookup(table) -> None:
    st.markdown(
        f"Table {table.number}: planned point → **{quantity_caption(table)}**. "
        "Type the rate, tubing-head pressure, water cut, GOR and lift the development "
        "strategy will use. A value outside the table is extrapolated by Eclipse, not clamped."
    )
    query: dict[str, float] = {}
    for name in AXIS_ORDER:
        values = table.axes[name].values
        if values.size == 1:
            st.caption(
                f"{axis_caption(table, name)} is a single value ({values[0]:g}). "
                "Eclipse treats the table as independent of it."
            )
            continue
        default = float(values[len(values) // 2])
        query[name] = float(
            st.number_input(
                axis_caption(table, name),
                value=default,
                key=f"lu_{table.number}_{name}",
                help=f"Table covers {values[0]:g} to {values[-1]:g} {table.axes[name].unit}".strip(),
            )
        )
    point = evaluate_operating_point(
        table,
        flo=query.get("FLO", float(table.axes["FLO"].values[0])),
        thp=query.get("THP", float(table.axes["THP"].values[0])),
        wfr=query.get("WFR"),
        gfr=query.get("GFR"),
        alq=query.get("ALQ"),
    )
    st.markdown(f"**{point.quantity}:** {point.linear_bhp:.6g} {point.unit}")
    if point.inside:
        st.success(point.note)
    else:
        st.markdown(
            f"Table-edge value: {point.edge_bhp:.6g} {point.unit} (not the Eclipse answer)."
        )
        st.warning(point.note)


def _fixed_sliders(table, exclude: set[str], key_prefix: str):
    fixed: dict[str, int] = {}
    for name in AXIS_ORDER:
        values = table.axes[name].values
        if values.size <= 1 or name in exclude:
            continue
        index = st.slider(
            axis_caption(table, name),
            min_value=0,
            max_value=values.size - 1,
            value=0,
            key=f"{key_prefix}_{table.number}_{name}",
            help=f"Index into {values[0]:g} … {values[-1]:g}",
        )
        fixed[name] = int(index)
    return fixed


def _axis_choices(table) -> list[str]:
    return [name for name in AXIS_ORDER if table.axis_lengths[name] > 1]


def _pick_axis(table, axes: list[str], preferred: str, label: str, key: str) -> str:
    index = axes.index(preferred) if preferred in axes else 0
    return st.selectbox(
        label,
        axes,
        index=index,
        format_func=lambda name: axis_caption(table, name),
        key=key,
    )


def _view_curves(findings, table) -> None:
    from vfpscope.viz.figures import apply_plot_hint, branch_figure, lift_curve_figure

    axes = _axis_choices(table)
    if len(axes) < 2:
        st.info("This table has fewer than two varying axes, so there is no curve family to draw.")
        return
    preset = st.selectbox("Preset", list(PRESETS), key=f"cv_preset_{table.number}")
    preferred_x, preferred_family = PRESETS[preset]
    x_axis = _pick_axis(table, axes, preferred_x, "x axis", f"cv_x_{table.number}")
    family_axes = [name for name in axes if name != x_axis]
    family = _pick_axis(
        table,
        family_axes,
        preferred_family,
        "curve family",
        f"cv_family_{table.number}",
    )
    fixed = _fixed_sliders(table, exclude={x_axis, family}, key_prefix="cv")
    show_hints = st.checkbox("Show QC plot hints", value=False, key=f"cv_hints_{table.number}")
    if table.role == "BRANCH":
        fig = branch_figure(table, x_axis=x_axis, family=family, fixed=fixed)
    else:
        fig = lift_curve_figure(table, x_axis=x_axis, family=family, fixed=fixed)
    if show_hints:
        for finding in findings:
            if finding.table_number == table.number:
                apply_plot_hint(fig, finding.plot_hint)
    st.caption(
        f"{quantity_caption(table)} for table {table.number}. "
        "The simulator interpolates between these curves."
    )
    _chart(fig)


def _view_qc(findings, table) -> None:
    rows = [
        finding
        for finding in findings
        if finding.table_number in {table.number, 0}
    ]
    if not rows:
        st.success("No QC findings for this table.")
        return
    order = {"ERROR": 2, "WARNING": 1, "INFO": 0}
    box = {"ERROR": st.error, "WARNING": st.warning, "INFO": st.info}
    for finding in sorted(rows, key=lambda item: -order[item.severity]):
        locus = f"\n\nLocus: `{finding.locus}`" if finding.locus else ""
        box[finding.severity](
            f"{finding.check_id} ({finding.severity}) — {finding.message}"
            f"{locus}\n\n{finding_implication(finding.check_id)}"
        )
    st.caption("Terminal gate: `vfpscope qc <deck> --fail-on warning`")


def _view_more(deck, table) -> None:
    heat, compare, coverage, network = st.tabs(["Heatmap", "Compare", "Coverage", "Network"])
    with heat:
        _view_heatmap(table)
    with compare:
        _view_compare(table)
    with coverage:
        _view_coverage(deck, table)
    with network:
        _view_network(deck)


def _view_heatmap(table) -> None:
    from vfpscope.viz.figures import heatmap_figure

    axes = _axis_choices(table)
    if len(axes) < 2:
        st.info("Need at least two varying axes for a heatmap.")
        return
    x_axis = _pick_axis(table, axes, axes[0], "heatmap x", f"hm_x_{table.number}")
    y_axes = [name for name in axes if name != x_axis]
    y_axis = _pick_axis(table, y_axes, y_axes[0], "heatmap y", f"hm_y_{table.number}")
    fixed = _fixed_sliders(table, exclude={x_axis, y_axis}, key_prefix="hm")
    as_contour = st.checkbox("Contour (vs 3-D surface)", value=True, key=f"hm_contour_{table.number}")
    _chart(
        heatmap_figure(
            table, x_axis=x_axis, y_axis=y_axis, fixed=fixed, as_contour=as_contour
        )
    )


def _view_compare(table) -> None:
    from vfpscope.viz.figures import compare_difference_figure, compare_figure

    st.markdown(
        "Overlay a second table from another Petrel export. "
        "Mismatched unit systems or axis types are rejected, not converted."
    )
    other = st.text_input("Second deck / include path", key="compare_path")
    if not other or not os.path.exists(other):
        st.info("Enter a second deck path to compare.")
        return
    try:
        other_deck = parse_deck(str(Path(other).resolve()))
    except VfpParseError as exc:
        st.error(str(exc))
        return
    opts = [f"#{number} {other_deck.tables[number].kind}" for number in other_deck.table_order]
    if not opts:
        st.info("That file has no VFP tables to compare.")
        return
    choice = st.selectbox("Second table", opts, key="compare_table")
    if not choice:
        st.info("That file has no VFP tables to compare.")
        return
    other_no = int(choice.split()[0].lstrip("#"))
    other_table = other_deck.tables[other_no]
    try:
        _chart(compare_figure([table, other_table]))
    except ValueError as exc:
        st.error(str(exc))
        return
    try:
        _chart(compare_difference_figure(table, other_table))
    except ValueError as exc:
        st.warning(f"Difference panel skipped: {exc}")


def _view_coverage(deck, table) -> None:
    from vfpscope.core.coverage import CoverageUnavailable, coverage_for_deck

    st.markdown(
        "After the Petrel run, point at the Eclipse `.UNSMRY`. "
        "Requires the `resdata` extra. Missing summary vectors are reported, not invented."
    )
    smry = st.text_input("UNSMRY path", key="coverage_smry")
    if not smry or not os.path.exists(smry):
        st.info("Enter a summary path to compute coverage.")
        return
    try:
        reports_by_table = coverage_for_deck(deck, smry)
    except CoverageUnavailable as exc:
        st.error(str(exc))
        return
    reports = reports_by_table.get(table.number, [])
    if not reports:
        st.info(f"No coverage vectors found for table {table.number} in that run.")
        return
    from vfpscope.viz.figures import coverage_overlay_figure

    for rep in reports:
        _chart(coverage_overlay_figure(table, rep))
        st.caption(
            "per-axis clamped fractions: "
            + ", ".join(f"{name} {rep.fraction_clamped(name) * 100:.1f}%" for name in rep.axes_with_data)
        )


def _view_network(deck) -> None:
    from vfpscope.viz.figures import network_graph_figure

    if not deck.branches:
        st.info("No BRANPROP branches in this deck.")
        return
    _chart(network_graph_figure(deck))
    st.markdown("**Branches**")
    for downtree, uptree, vfp in deck.branches:
        table = deck.tables.get(vfp)
        if table:
            top = table.axes["FLO"].values[-1]
            label = f"VFP {vfp} — {axis_caption(table, 'FLO')} up to {top:g}"
        else:
            label = f"VFP {vfp} (missing)"
        st.markdown(f"- **{downtree} → {uptree}**: {label}")


if __name__ == "__main__":
    main()
