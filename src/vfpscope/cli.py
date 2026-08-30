"""VFPScope CLI — Typer entry point.

Commands:
  list    - every VFP table in a deck/include, with role, axes, consumers
  qc      - QC findings; non-zero exit when a severity threshold is hit
  serve   - launch the Streamlit GUI
  plot    - export a table's lift curve to PNG/HTML
  report  - self-contained HTML report of all tables + QC verdicts
"""

from __future__ import annotations

from pathlib import Path

import typer

from .core.model import SEVERITY_ORDER
from .core.parse.native import VfpParseError, load_deck

app = typer.Typer(help="VFPScope — interactive VFP table visualizer for Eclipse/OPM decks")


@app.command("list")
def list_tables(deck: Path = typer.Argument(..., help=".DATA deck or VFP include file")):
    """List every VFP table in the deck with role, axis types and consumers."""
    d = _load(deck)
    hdr = (
        f"{'#':>3}  {'KIND':<8} {'ROLE':<8} {'FLO':<5} {'WFR':<5} {'GFR':<5} "
        f"{'ALQ':<5} {'UNITS':<8} {'NTHP':>4} {'NWFR':>4} {'NGFR':>4} {'NALQ':>4} "
        f"{'NFLO':>4}  CONSUMERS"
    )
    typer.echo(f"VFP tables in {deck}")
    typer.echo(hdr)
    typer.echo("-" * len(hdr))
    for no in d.table_order:
        t = d.tables[no]
        n = t.axis_lengths
        consumers = []
        if t.consumers.wells:
            consumers.append("wells: " + ", ".join(t.consumers.wells))
        if t.consumers.branches:
            consumers.append("branches: " + ", ".join(f"{a}->{b}" for a, b in t.consumers.branches))
        if not consumers:
            consumers.append("(none)")
        typer.echo(
            f"{no:>3}  {t.kind:<8} {t.role:<8} {t.axes['FLO'].kind:<5} "
            f"{t.axes['WFR'].kind:<5} {t.axes['GFR'].kind:<5} {t.axes['ALQ'].kind or '':<5} "
            f"{t.unit_system:<8} {n['THP']:>4} {n['WFR']:>4} {n['GFR']:>4} {n['ALQ']:>4} "
            f"{n['FLO']:>4}  {'; '.join(consumers)}"
        )
    n_err = sum(1 for f in d.findings if f.severity == "ERROR")
    n_warn = sum(1 for f in d.findings if f.severity == "WARNING")
    n_info = sum(1 for f in d.findings if f.severity == "INFO")
    typer.echo(
        f"{len(d.tables)} table(s), {n_err} error(s), {n_warn} warning(s), {n_info} info(s)"
    )
    raise typer.Exit(0)


@app.command("qc")
def qc(
    deck: Path = typer.Argument(..., help=".DATA deck or VFP include file"),
    fail_on: str = typer.Option(
        "error", "--fail-on", help="Exit non-zero when a finding of this severity or worse exists"
    ),
):
    """Run QC checks; exit 0 (pass) or 1 (fail)."""
    from rich.console import Console

    console = Console()
    fail_on = fail_on.upper()
    if fail_on not in SEVERITY_ORDER:
        console.print(f"[red]--fail-on must be one of {list(SEVERITY_ORDER)}[/]")
        raise typer.Exit(2)
    d = _load(deck)
    from .core.qc.engine import run_qc

    findings = run_qc(d)
    findings = sorted(
        findings,
        key=lambda f: (-SEVERITY_ORDER[f.severity], f.table_number, f.check_id),
    )
    for f in findings:
        color = {"ERROR": "red", "WARNING": "yellow", "INFO": "dim"}[f.severity]
        loc = f.locus.get("path", "")
        console.print(
            f"[{color}]{f.severity:<7}[/] [{color}]{f.check_id}[/] "
            f"table {f.table_number}: {f.message}"
            + (f"  [{color}]({loc})[/]" if loc else "")
        )
    threshold = SEVERITY_ORDER[fail_on]
    worst = max((SEVERITY_ORDER[f.severity] for f in findings), default=0)
    summary = (
        f"[bold]{len(d.tables)} table(s)[/], {len(findings)} finding(s), "
        f"worst severity: {next((s for s, v in sorted(SEVERITY_ORDER.items(), key=lambda kv: -kv[1]) if v <= worst), 'none')}"
    )
    console.print(summary)
    if worst >= threshold:
        raise typer.Exit(1)
    raise typer.Exit(0)


@app.command("serve")
def serve(
    deck: Path = typer.Argument(..., help=".DATA deck or VFP include file to open in the GUI"),
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8501, "--port"),
):
    """Launch the Streamlit GUI on the given deck."""
    import subprocess
    import sys

    app_py = Path(__file__).resolve().parent / "app" / "main.py"
    if not app_py.exists():
        typer.echo(f"GUI entry point not found: {app_py}", err=True)
        raise typer.Exit(2)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_py),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--",
        "--deck",
        str(deck.resolve()),
    ]
    raise typer.Exit(subprocess.call(cmd))


@app.command("plot")
def plot(
    deck: Path = typer.Argument(..., help=".DATA deck or VFP include file"),
    table_no: int = typer.Option(1, "--table", "-t", help="table number to plot"),
    png: Path | None = typer.Option(None, "--png", help="write PNG (needs kaleido)"),
    html: Path | None = typer.Option(None, "--html", help="write self-contained HTML"),
):
    """Export one table's lift curve as PNG and/or HTML."""
    from .viz.figures import lift_curve_figure

    d = _load(deck)
    if table_no not in d.tables:
        typer.echo(f"table {table_no} not found in {deck}", err=True)
        raise typer.Exit(2)
    t = d.tables[table_no]
    fig = lift_curve_figure(t)
    wrote = []
    if png:
        try:
            fig.write_image(str(png), width=1200, height=700)
            wrote.append(str(png))
        except Exception as e:  # kaleido missing / renderer failure
            typer.echo(f"PNG export failed: {e} (pip install kaleido)", err=True)
            raise typer.Exit(2) from None
    if html:
        fig.write_html(str(html), include_plotlyjs="cdn")
        wrote.append(str(html))
    if not wrote:
        fig.write_html(str(deck.with_suffix(".vfp_plot.html")), include_plotlyjs="cdn")
        wrote.append(str(deck.with_suffix(".vfp_plot.html")))
    typer.echo("wrote: " + ", ".join(wrote))
    raise typer.Exit(0)


@app.command("report")
def report(
    deck: Path = typer.Argument(..., help=".DATA deck or VFP include file"),
    out: Path = typer.Option("report.html", "-o", "--out", help="output HTML path"),
):
    """Write a self-contained HTML report: every table's curves + QC verdict."""
    from .viz.figures import build_report_html

    d = _load(deck)
    build_report_html(d, out)
    typer.echo(f"wrote {out}")
    raise typer.Exit(0)


def _load(deck: Path):
    try:
        return load_deck(deck)
    except VfpParseError as e:
        typer.echo(f"[red]{e}[/]", err=True)
        raise typer.Exit(2) from None
    except OSError as e:
        typer.echo(f"cannot read {deck}: {e}", err=True)
        raise typer.Exit(2) from None


if __name__ == "__main__":
    app()
