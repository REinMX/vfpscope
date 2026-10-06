# VFPScope

Interactive VFP check for a Petrel simulation case.

Reads `VFPPROD` / `VFPINJ` from the Eclipse deck Petrel exported, shows which
well or branch uses each table, and gates the run before you press Simulate.

## Quick start

```
uv sync --extra test
uv run vfpscope list deck.DATA
uv run vfpscope qc deck.DATA --fail-on warning
uv run vfpscope serve deck.DATA      # Petrel simulation GUI
uv run vfpscope report deck.DATA -o report.html
```

`serve` opens on Case: well → table, a run gate, and a CSV of the assignments.
Lookup evaluates one development-strategy point. Outside the table, Eclipse
extrapolates; the screen says so and does not treat the table-edge pressure
as the simulator answer.

The single-file edition below is the restricted parser/QC copy. It does not
include this Petrel workspace.

## Single-file edition

For restricted environments where copying a package tree is inconvenient, use
[`standalone/vfpscope_standalone.py`](standalone/vfpscope_standalone.py). It is a
self-contained source file and does not import the `vfpscope` package.

Copy or download that one file into Python 3.11+, verify the five runtime
dependencies already exist in the target virtual environment, and launch it:

```
python -c "import numpy, pydantic, plotly, streamlit, resdata; print('VFPScope dependencies available')"
streamlit run vfpscope_standalone.py -- --deck MODEL.DATA
```

The standalone edition includes native deck/include parsing, consumer roles,
QC, curves, heatmaps, comparison, network views and Eclipse/OPM `.UNSMRY`
operating-envelope coverage through `resdata`.

Direct download:

```
https://raw.githubusercontent.com/REinMX/vfpscope/main/standalone/vfpscope_standalone.py
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
