# VFPScope implementation guide

This guide explains the design and the safest way to extend the code.

## Purpose

VFPScope reads and evaluates existing Eclipse/OPM VFP tables. It does not generate
new tables from multiphase-flow correlations.

The core contract is:

```text
VFPPROD: BHP = f(FLO, THP, WFR, GFR, ALQ)
VFPINJ:  BHP = f(FLO, THP)
```

All table data uses one canonical NumPy order:

```text
(THP, WFR, GFR, ALQ, FLO)
```

Never change this order in individual modules. Deck indices are converted from
1-based to 0-based once, at the parser boundary.

## Architecture

```text
src/vfpscope/
├── core/
│   ├── model.py          Frozen models and axis/unit definitions
│   ├── parse/native.py   Dependency-free Eclipse parser
│   ├── refs.py           Well/branch references and role inference
│   ├── interp.py         Multilinear lookup with edge clamping
│   ├── derive.py         Delta-P, gradients and turning points
│   ├── coverage.py       UNSMRY operating-envelope analysis
│   └── qc/               Pure QC checks and registry
├── viz/figures.py        Pure Plotly figure builders
├── app/main.py           Streamlit UI
└── cli.py                Typer CLI
```

Architectural rules:

1. `core/` must not import Streamlit or Plotly.
2. Figure functions return `go.Figure`; they never render it.
3. QC checks return structured `Finding` objects; they never print.
4. Broken input is reported, never silently repaired.
5. Parser errors include source file and line number.

## Native parser flow

`load_deck(path)` performs these steps:

1. Tokenize Eclipse free-format text.
2. Expand repeat syntax such as `5*0.0`; preserve `1*` as a default.
3. Traverse `INCLUDE` files relative to the including file and resolve `PATHS`
   aliases.
4. Parse `VFPPROD` and `VFPINJ` headers, axes and data records.
5. Validate record count, row width, indices, enums and axis ordering.
6. Scan `WCONPROD`, `WCONINJE`, `BRANPROP`, `NODEPROP`, `WELSPECS` and
   `WLIFTOPT`.
7. Infer each table role: `WELL`, `BRANCH` or `UNKNOWN`.

The parser mirrors the OPM/res2df `VFPINJ` header layout:

```text
TABLE DATUM_DEPTH RATE_TYPE PRESSURE_DEF UNITS BODY_DEF
```

Real files may provide only the first three items and default the rest.

## Adding a QC check

Use strict test-driven development:

1. Add a deliberately broken fixture or programmatic table.
2. Write a failing test in `tests/test_qc.py`.
3. Implement a pure function in `core/qc/checks.py`:

```python
def check_example(table: VfpTable, ctx: QCContext) -> list[Finding]:
    if condition_is_ok:
        return []
    return [Finding(
        check_id="EXAMPLE",
        severity="WARNING",
        table_number=table.number,
        message="Engineering explanation with values",
        locus={"thp": 0, "wfr": 1},
        plot_hint={"kind": "example", "slice": [0, 1, 0, 0]},
    )]
```

4. Register it in `core/qc/engine.py`.
5. Add UI highlighting in `viz/figures.py` when a new `plot_hint.kind` is needed.
6. Run the full suite.

## Adding a view

Keep data extraction separate from rendering:

1. Put calculations in `core/derive.py` or another `core/` module.
2. Add a pure figure builder to `viz/figures.py`.
3. Test traces, axes and values without starting Streamlit.
4. Wrap the figure in a small function in `app/main.py`.

## Optional dependency boundaries

- `res2df`: parser cross-check and future deck export; not required at runtime.
- `resdata`: `.UNSMRY` coverage analysis; install with the `resdata` extra.
- Chrome: required by Kaleido for PNG export. HTML export does not need Chrome.

## Development commands

```bash
git clone git@github.com:REinMX/vfpscope.git
cd vfpscope
uv sync --extra test --extra res2df --extra resdata

uv run pytest
uv run ruff check src tests
uv run vfpscope --help
```

Useful focused tests:

```bash
uv run pytest tests/test_tokenizer.py tests/test_parse_vfpprod.py
uv run pytest tests/test_interp.py tests/test_qc.py
uv run pytest tests/test_app.py
```

## Acceptance checklist

Before committing a change:

- Parser arrays remain in `(THP, WFR, GFR, ALQ, FLO)` order.
- Structural failures remain fail-closed and source-located.
- Lookup clamps edges and reports clamping; it never extrapolates.
- WELL and BRANCH labels/derived quantities remain physically distinct.
- New QC findings include a useful `locus` and, where visual, `plot_hint`.
- `uv run pytest` and `uv run ruff check src tests` pass.
