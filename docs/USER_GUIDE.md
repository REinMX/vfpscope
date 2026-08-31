# VFPScope user guide

VFPScope is an interactive viewer and QC gate for existing Eclipse/OPM
`VFPPROD` and `VFPINJ` tables. It supports well lift tables and network branch
pressure-drop tables.

## Supported input

You can load:

- An Eclipse/OPM `.DATA` deck.
- A bare include file containing `VFPPROD` or `VFPINJ`.
- Text exports with extensions such as `.inc`, `.ecl` or `.VFP`, provided the
  contents use Eclipse `VFPPROD`/`VFPINJ` syntax.
- A PROSPER table exported in Eclipse VFP format.
- An Eclipse/OPM `.UNSMRY` result for operating-envelope coverage, when the
  corresponding deck identifies the consuming wells.

You cannot load a native PROSPER `.CFP` project directly. Export its VFP table
from PROSPER in Eclipse format first. VFPScope reads and judges tables; it does
not reproduce PROSPER correlations or generate tables.

## Install

```bash
git clone git@github.com:REinMX/vfpscope.git
cd vfpscope
uv sync --extra test --extra resdata
```

For only the normal GUI and CLI, this is sufficient:

```bash
uv sync
```

## Inspect a deck from the terminal

List every table, its axis types, role and consumers:

```bash
uv run vfpscope list MODEL.DATA
```

Run automatic QC:

```bash
uv run vfpscope qc MODEL.DATA
```

Use it as a pre-run gate:

```bash
uv run vfpscope qc MODEL.DATA --fail-on warning
```

Exit codes:

- `0`: no finding at or above the chosen threshold.
- `1`: QC gate failed.
- `2`: bad command, unreadable file or parser-level failure.

## Open the GUI

```bash
uv run vfpscope serve MODEL.DATA
```

The main views are:

- **Curves:** BHP vs FLO by THP, or any other axis pivot.
- **Heatmap:** tabulated pressure over any two non-degenerate axes.
- **QC:** structural, physical, consistency and coverage findings.
- **Compare:** overlay two tables; mismatched unit systems or axis types are
  rejected rather than silently converted.
- **Coverage:** overlay run operating points from `.UNSMRY` and quantify
  clamped timesteps by axis.
- **Network:** display `BRANPROP`/`NODEPROP` topology.

For a remote Linux host, bind explicitly only on a trusted private network:

```bash
uv run vfpscope serve MODEL.DATA --host 0.0.0.0 --port 8501
```

Prefer SSH forwarding or Tailscale instead of exposing Streamlit publicly.

## Export plots and reports

Self-contained HTML report for all tables:

```bash
uv run vfpscope report MODEL.DATA -o vfp_report.html
```

One table as interactive HTML:

```bash
uv run vfpscope plot MODEL.DATA --table 3 --html table_3.html
```

PNG export:

```bash
uv run vfpscope plot MODEL.DATA --table 3 --png table_3.png
```

PNG requires Chrome because Plotly/Kaleido uses it. On a headless machine
without Chrome, use HTML.

## Bare include files

The same commands work on an include file:

```bash
uv run vfpscope list well_vfp.inc
uv run vfpscope serve well_vfp.inc
uv run vfpscope report well_vfp.inc -o report.html
```

A table in a bare include has role `UNKNOWN` unless a parent deck supplies
`WCONPROD`, `WCONINJE` or `BRANPROP` references.

## PROSPER workflow

1. Open the well or pipeline model in PROSPER.
2. Export the calculated VFP table in Eclipse format.
3. Confirm the output contains `VFPPROD` or `VFPINJ` text.
4. Load that exported file directly:

```bash
uv run vfpscope serve exported_table.VFP
```

A native `.CFP` file is not accepted directly.

## Coverage against a simulation run

Install the optional reader:

```bash
uv sync --extra resdata
```

Start VFPScope with the deck, open **Coverage**, and enter the `.UNSMRY` path.
VFPScope maps table axis types to well summary vectors such as `WTHP`, `WOPR`,
`WWPR`, `WGPR`, `WWCT` and `WGOR`.

The output reports:

- Fraction of timesteps clamped low or high on each axis.
- Run maximum divided by table maximum.
- Operating points lying on an unstable branch.

If required summary vectors were not requested in the simulation, VFPScope
reports that coverage is unavailable rather than inventing values.

## Interpreting roles

- `WELL`: referenced from `WCONPROD` or `WCONINJE`; the tabulated quantity is
  treated as bottom-hole pressure and THP as wellhead pressure.
- `BRANCH`: referenced from `BRANPROP`; the tabulated quantity is inlet pressure,
  THP represents outlet pressure, and VFPScope also plots `delta-P`.
- `UNKNOWN`: no consumer was found in the loaded input.

A table used by both a well and a branch receives a `ROLE_CONFLICT` warning.

## Common errors

- **Expected N data records, found M:** truncated, duplicated or malformed table.
- **Axis not strictly increasing:** duplicate or out-of-order axis points.
- **THP_MONOTONIC/CROSSING:** pressure response is inconsistent across THP curves.
- **UNSTABLE_BRANCH:** BHP vs FLO contains a turning point; this can create dual
  solutions or solver oscillation.
- **CLAMP_FRACTION:** the run operates outside table coverage and the simulator
  repeatedly uses an edge value.
- **UNITS_MISMATCH:** table and deck unit systems disagree.

VFPScope does not silently repair these conditions. Correct the source table or
regenerate/export it from the original engineering tool.
