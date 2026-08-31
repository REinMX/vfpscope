# VFPScope

Interactive VFP table visualizer for Eclipse / OPM Flow decks.

Reads `VFPPROD` / `VFPINJ` tables from `.DATA` decks or bare include files,
labels them by table number and by the wells/branches that consume them,
and provides interactive exploration, automatic QC, coverage analysis and
table comparison — for both well (tubing lift) tables and network branch
(flowline) tables.

## Quick start

```
uv sync --extra test
uv run vfpscope list deck.DATA
uv run vfpscope qc deck.DATA --fail-on warning
uv run vfpscope serve deck.DATA      # Streamlit GUI
uv run vfpscope report deck.DATA -o report.html
```

## Input formats

VFPScope reads Eclipse/OPM `.DATA` decks and text include files containing
`VFPPROD` or `VFPINJ` (`.inc`, `.ecl`, `.VFP`). PROSPER tables work when
exported in Eclipse VFP format; native PROSPER `.CFP` projects are not read
directly. Optional `.UNSMRY` input provides operating-envelope coverage.

## Guides

- [User guide](docs/USER_GUIDE.md) — install, supported formats, GUI/CLI,
  PROSPER export workflow, coverage and troubleshooting.
- [Implementation guide](docs/IMPLEMENTATION_GUIDE.md) — architecture, parser
  invariants, adding QC checks/views, tests and acceptance rules.

## Layout

- `src/vfpscope/core/` — pure library: model, parser (`parse/native.py`),
  deck references, interpolation, derivation, QC engine. No UI imports.
- `src/vfpscope/viz/figures.py` — Plotly figure builders (no rendering).
- `src/vfpscope/app/` — Streamlit GUI.
- `src/vfpscope/cli.py` — Typer CLI.

Managed with `uv` (pyproject is PEP 621, Poetry-compatible).
