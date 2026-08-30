"""Native Eclipse free-format deck/include parser — no OPM dependency.

Handles: ``--`` comments, ``/`` record terminators, ``n*value`` repeat counts,
``1*`` / ``*`` defaults, quoted strings (``--`` and ``/`` inside quotes are
literal), ``INCLUDE`` resolution (relative to the including file, plus
``PATHS`` aliases), and line continuation across records.

Structural rules enforced here (never silently repaired):

* VFPPROD must contain exactly ``NTHP x NWFR x NGFR x NALQ`` data records,
  each with exactly ``NFLO`` values and in-range 1-based indices.
* Axis vectors must be non-empty, numeric and strictly increasing.
* A record-count mismatch raises a located ``VfpParseError``
  ("VFPPROD 3: expected 24 data records, found 23").

``load_deck`` converts per-table parse failures into ERROR findings so the
rest of the deck still loads; ``parse_vfp_file`` re-raises them (single-file
mode used by tests and include inspection).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model import (
    ALQ_TYPES,
    AXIS_ORDER,
    GFR_TYPES,
    TABULATED_TYPES,
    THP_TYPES,
    UNIT_SYSTEMS,
    VFPINJ_FLO_TYPES,
    VFPPROD_FLO_TYPES,
    WFR_TYPES,
    Finding,
    SourceRef,
    VfpAxis,
    VfpTable,
    resolve_axis_unit,
)

_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([EeDd][+-]?\d+)?$")
_REPEAT_RE = re.compile(r"^(\d+)\*(.*)$")
_MAX_INCLUDE_DEPTH = 50


class VfpParseError(Exception):
    """Located parse error. ``str(e)`` carries file:line; ``e.raw`` is bare."""

    def __init__(self, message: str, path: str | None = None, line: int | None = None):
        self.path = path
        self.line = line
        self.raw = message
        loc = f"{path}:{line}: " if path and line else ""
        super().__init__(f"{loc}{message}")


# --------------------------------------------------------------------------- tokenizer


@dataclass(frozen=True)
class Token:
    text: str
    file: str
    line: int
    is_record_end: bool = False
    quoted: bool = False


def tokenize(text: str, path: str) -> list[Token]:
    """Split deck text into tokens, handling quotes, comments and `/`."""
    toks: list[Token] = []
    i, n = 0, len(text)
    line = 1
    while i < n:
        c = text[i]
        if c in " \t\r":
            i += 1
            continue
        if c == "\n":
            line += 1
            i += 1
            continue
        if c == "-" and text[i : i + 2] == "--":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "'":
            j = i + 1
            buf: list[str] = []
            closed = False
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    closed = True
                    break
                buf.append(text[j])
                j += 1
            if not closed:
                raise VfpParseError("unterminated quoted string", path, line)
            toks.append(Token("".join(buf), path, line, quoted=True))
            i = j + 1
            continue
        if c == "/":
            toks.append(Token("/", path, line, is_record_end=True))
            i += 1
            continue
        # plain token: read until whitespace, `/`, quote or comment start.
        # A repeat count directly followed by a quoted string (`3*'GAS'`) is
        # consumed as one token so expansion sees `3*GAS`.
        j = i
        while j < n:
            ch = text[j]
            if ch in " \t\r\n/'" or (ch == "-" and text[j : j + 2] == "--"):
                break
            j += 1
        if j < n and text[j] == "'" and _REPEAT_RE.match(text[i:j]):
            # consume the quoted string and append its content
            k = j + 1
            buf = [text[i:j]]
            closed = False
            while k < n:
                if text[k] == "'":
                    if k + 1 < n and text[k + 1] == "'":
                        buf.append("'")
                        k += 2
                        continue
                    closed = True
                    break
                buf.append(text[k])
                k += 1
            if not closed:
                raise VfpParseError("unterminated quoted string", path, line)
            toks.append(Token("".join(buf), path, line))
            i = k + 1
            continue
        toks.append(Token(text[i:j], path, line))
        i = j
    return toks


def parse_string(text: str, path: str = "<string>") -> list[Token]:
    return tokenize(text, path)


def _is_numeric(text: str) -> bool:
    return bool(_NUM_RE.match(text))


def _as_float(text: str) -> float | None:
    """Parse a numeric token; None if it is not numeric."""
    if _is_numeric(text):
        return float(text.replace("D", "E").replace("d", "e"))
    return None


@dataclass(frozen=True)
class Item:
    value: float | str | None  # None == default (1*, *)
    file: str
    line: int

    @property
    def is_default(self) -> bool:
        return self.value is None

    def as_int(self, what: str) -> int:
        if self.value is None:
            raise VfpParseError(f"missing (default) value for {what}", self.file, self.line)
        if not isinstance(self.value, float) or not float(self.value).is_integer():
            raise VfpParseError(
                f"{what}: expected an integer, got {self.value!r}", self.file, self.line
            )
        return int(self.value)

    def as_float(self, what: str) -> float:
        if self.value is None:
            raise VfpParseError(f"missing (default) value for {what}", self.file, self.line)
        if not isinstance(self.value, float):
            raise VfpParseError(
                f"{what}: expected a number, got {self.value!r}", self.file, self.line
            )
        return self.value

    def as_str(self, what: str) -> str:
        if self.value is None:
            raise VfpParseError(f"missing (default) value for {what}", self.file, self.line)
        return str(self.value)


@dataclass(frozen=True)
class Record:
    items: tuple[Item, ...]
    file: str
    line: int


def expand_record_items(tokens: Iterable[Token]) -> list[Item]:
    """Expand ``n*value`` / ``1*`` tokens into flat items (defaults -> None)."""
    items: list[Item] = []
    for tok in tokens:
        if tok.is_record_end:
            continue
        if tok.quoted:
            items.append(Item(tok.text, tok.file, tok.line))
            continue
        text = tok.text
        if text == "*":
            items.append(Item(None, tok.file, tok.line))
            continue
        m = _REPEAT_RE.match(text)
        if m:
            count = int(m.group(1))
            rest = m.group(2)
            if not rest:  # "5*" -> five defaults
                items.extend(Item(None, tok.file, tok.line) for _ in range(count))
                continue
            value: float | str
            f = _as_float(rest)
            if f is None:
                value = rest
            else:
                value = f
            items.extend(Item(value, tok.file, tok.line) for _ in range(count))
            continue
        f = _as_float(text)
        items.append(Item(f if f is not None else text, tok.file, tok.line))
    return items


def _read_record(tokens: list[Token], pos: int) -> tuple[Record | None, int]:
    """Read one record (up to and including `/`) starting at ``pos``."""
    if pos >= len(tokens):
        return None, pos
    start = pos
    while pos < len(tokens) and not tokens[pos].is_record_end:
        pos += 1
    if pos >= len(tokens):
        return None, start  # unterminated at EOF; caller reports
    rec_tokens = tokens[start:pos]
    items = expand_record_items(rec_tokens)
    file = rec_tokens[0].file if rec_tokens else tokens[start].file
    line = rec_tokens[0].line if rec_tokens else tokens[start].line
    return Record(tuple(items), file, line), pos + 1


# --------------------------------------------------------------------------- deck scan


@dataclass
class Deck:
    """Everything extracted from one deck/include tree."""

    path: str
    unit_system: str = "DEFAULT"  # METRIC/FIELD/LAB/PVT-M from RUNSPEC, else DEFAULT
    tables: dict[int, VfpTable] = field(default_factory=dict)
    table_order: list[int] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    # --- reference maps (refs.py turns these into roles/consumers) ---
    well_vfp: dict[str, int] = field(default_factory=dict)  # WELL -> table no
    branch_vfp: dict[tuple[str, str], int] = field(default_factory=dict)  # (dt,ut) -> table
    branches: list[tuple[str, str, int]] = field(default_factory=list)
    nodes: list[tuple[str, float | None]] = field(default_factory=list)  # (node, pressure)
    wells_datum: dict[str, float] = field(default_factory=dict)
    well_phases: dict[str, str] = field(default_factory=dict)
    gaslift_wells: set[str] = field(default_factory=set)

    def finding(
        self,
        check_id: str,
        severity: str,
        table_number: int,
        message: str,
        locus: dict | None = None,
        plot_hint: dict | None = None,
    ) -> None:
        self.findings.append(
            Finding(
                check_id=check_id,
                severity=severity,  # type: ignore[arg-type]
                table_number=table_number,
                message=message,
                locus=locus or {},
                plot_hint=plot_hint,
            )
        )


_INTEREST_KEYWORDS = {
    "VFPPROD",
    "VFPINJ",
    "INCLUDE",
    "PATHS",
    "WCONPROD",
    "WCONINJE",
    "BRANPROP",
    "NODEPROP",
    "WELSPECS",
    "WLIFTOPT",
    "METRIC",
    "FIELD",
    "LAB",
    "PVT-M",
}
_UNIT_KEYWORDS = {"METRIC": "METRIC", "FIELD": "FIELD", "LAB": "LAB", "PVT-M": "PVT-M"}


def _find_keyword_start(tokens: list[Token], pos: int) -> int:
    """Next token that is a bare (unquoted, non-numeric) identifier."""
    while pos < len(tokens):
        t = tokens[pos]
        if not t.is_record_end and not t.quoted and not _is_numeric(t.text):
            return pos
        pos += 1
    return pos


def load_deck(path: str | Path) -> Deck:
    """Parse a deck (or bare include) into a Deck with tables + reference maps.

    Per-table parse failures become ERROR findings (check_id PARSE); the rest
    of the deck still loads. Missing includes raise VfpParseError.
    """
    root = str(Path(path).resolve())
    deck = Deck(path=root)
    tokens = tokenize(Path(root).read_text(encoding="utf-8", errors="replace"), root)
    aliases: dict[str, str] = {}

    _scan_tokens(deck, tokens, 0, [root], aliases)

    # two-pass: assign roles/consumers now that every reference is collected
    from ..refs import assign_roles  # lazy: refs imports model only, avoids cycle

    assign_roles(deck)
    return deck


def _scan_tokens(
    deck: Deck,
    tokens: list[Token],
    pos: int,
    include_stack: list[str],
    aliases: dict[str, str],
) -> None:
    """Scan a token list for keywords; INCLUDEs recurse (stack = recursion)."""
    while True:
        pos = _find_keyword_start(tokens, pos)
        if pos >= len(tokens):
            return
        kw = tokens[pos].text.upper()
        pos += 1
        if kw == "VFPPROD":
            _parse_vfp_keyword(deck, "VFPPROD", tokens, pos)
            pos = _find_keyword_start(tokens, pos)
            continue
        if kw == "VFPINJ":
            _parse_vfp_keyword(deck, "VFPINJ", tokens, pos)
            pos = _find_keyword_start(tokens, pos)
            continue
        if kw == "INCLUDE":
            rec, pos = _read_record(tokens, pos)
            if rec is None:
                raise VfpParseError(
                    "INCLUDE without a record", tokens[pos - 1].file, tokens[pos - 1].line
                )
            for item in rec.items:
                name = item.as_str("INCLUDE filename")
                resolved = _resolve_include(name, item.file, aliases)
                abs_path = str(Path(resolved).resolve())
                if abs_path in include_stack:
                    raise VfpParseError(
                        f"include cycle detected: {abs_path}", item.file, item.line
                    )
                if len(include_stack) >= _MAX_INCLUDE_DEPTH:
                    raise VfpParseError(
                        "include depth limit exceeded", item.file, item.line
                    )
                if not Path(abs_path).exists():
                    raise VfpParseError(
                        f"include file not found: {name} (resolved {abs_path})",
                        item.file,
                        item.line,
                    )
                sub = tokenize(
                    Path(abs_path).read_text(encoding="utf-8", errors="replace"),
                    abs_path,
                )
                _scan_tokens(deck, sub, 0, include_stack + [abs_path], aliases)
            continue
        if kw == "PATHS":
            while True:
                rec, pos = _read_record(tokens, pos)
                if rec is None or not rec.items:
                    if rec is None:
                        raise VfpParseError(
                            "unterminated PATHS block (missing blank '/' record)",
                            tokens[pos - 1].file,
                            tokens[pos - 1].line,
                        )
                    break
                for i in range(0, len(rec.items) - 1, 2):
                    name = rec.items[i].as_str("PATHS alias name")
                    target = rec.items[i + 1].as_str("PATHS alias path")
                    aliases[name.upper()] = target
            continue
        if kw in _UNIT_KEYWORDS:
            deck.unit_system = kw
            continue
        if kw in ("WCONPROD", "WCONINJE", "BRANPROP", "NODEPROP", "WELSPECS", "WLIFTOPT"):
            pos = _consume_ref_keyword(deck, kw, tokens, pos)
            continue
        # unknown keyword: ignore its tokens entirely


def _resolve_include(name: str, file: str, aliases: dict[str, str]) -> str:
    base = Path(file).parent
    if name.startswith("$"):
        var, _, rest = name[1:].partition("/")
        if var.upper() not in aliases:
            raise VfpParseError(
                f"unknown PATHS alias '${var}' in include {name!r}", file, 0
            )
        return str(Path(aliases[var.upper()]) / rest)
    return str(base / name)


def _consume_ref_keyword(deck: Deck, kw: str, tokens: list[Token], pos: int) -> int:
    """Collect records for a schedule/reference keyword until the blank record."""
    n_records = 0
    while True:
        rec, pos = _read_record(tokens, pos)
        if rec is None:
            raise VfpParseError(
                f"unterminated {kw} block (missing blank '/' record)",
                tokens[pos - 1].file if pos else deck.path,
                tokens[pos - 1].line if pos else 0,
            )
        if not rec.items:
            return pos
        n_records += 1
        try:
            if kw == "WCONPROD":
                well = rec.items[0].as_str("well name").upper()
                if len(rec.items) > 10 and not rec.items[10].is_default:
                    vfp = rec.items[10].as_int(f"WCONPROD VFP table for {well}")
                    if vfp > 0:
                        deck.well_vfp[well] = vfp
            elif kw == "WCONINJE":
                well = rec.items[0].as_str("well name").upper()
                if len(rec.items) > 8 and not rec.items[8].is_default:
                    vfp = rec.items[8].as_int(f"WCONINJE VFP table for {well}")
                    if vfp > 0:
                        deck.well_vfp[well] = vfp
            elif kw == "BRANPROP":
                dt = rec.items[0].as_str("branch downtree node").upper()
                ut = rec.items[1].as_str("branch uptree node").upper()
                vfp = rec.items[2].as_int(f"BRANPROP VFP table for {dt}-{ut}")
                if vfp > 0:
                    deck.branch_vfp[(dt, ut)] = vfp
                    deck.branches.append((dt, ut, vfp))
            elif kw == "NODEPROP":
                name = rec.items[0].as_str("node name").upper()
                pressure = (
                    rec.items[1].as_float(f"NODEPROP pressure for {name}")
                    if len(rec.items) > 1 and not rec.items[1].is_default
                    else None
                )
                deck.nodes.append((name, pressure))
            elif kw == "WELSPECS":
                well = rec.items[0].as_str("well name").upper()
                if len(rec.items) > 4 and not rec.items[4].is_default:
                    deck.wells_datum[well] = rec.items[4].as_float(f"datum for {well}")
                if len(rec.items) > 5 and not rec.items[5].is_default:
                    deck.well_phases[well] = rec.items[5].as_str(f"phase for {well}").upper()
            elif kw == "WLIFTOPT":
                deck.gaslift_wells.add(rec.items[0].as_str("well name").upper())
        except VfpParseError as e:
            deck.finding(
                "PARSE", "ERROR", 0, f"in {kw}: {e}",
                locus={"path": e.path, "line": e.line},
            )
    return pos  # pragma: no cover


# --------------------------------------------------------------------------- VFP table parsing


def _parse_vfp_keyword(deck: Deck, kind: str, tokens: list[Token], pos: int) -> None:
    try:
        table, pos = _parse_table(deck, kind, tokens, pos)
    except VfpParseError as e:
        deck.finding(
            "PARSE",
            "ERROR",
            getattr(e, "table_number", 0) or 0,
            getattr(e, "raw", str(e)),
            locus={"path": e.path, "line": e.line},
        )
        # consume the rest of the broken block so scanning can continue
        _skip_to_next_keyword(tokens, pos)
        return
    if table.number in deck.tables:
        deck.finding(
            "DUP_TABLE",
            "ERROR",
            table.number,
            f"duplicate VFP table number {table.number} in deck "
            f"(first occurrence at {deck.tables[table.number].source})",
            locus={"path": table.source.path, "line": table.source.line},
        )
        return
    deck.tables[table.number] = table
    deck.table_order.append(table.number)


def _skip_to_next_keyword(tokens: list[Token], pos: int) -> None:
    # after a broken table, jump to the next bare identifier token
    _find_keyword_start(tokens, pos)


def _parse_table(deck: Deck, kind: str, tokens: list[Token], pos: int) -> tuple[VfpTable, int]:
    """Parse one VFPPROD/VFPINJ keyword block; returns (table, next_pos)."""
    kw_token = tokens[pos - 1]
    file, line = kw_token.file, kw_token.line

    # ---- header ------------------------------------------------------------
    header, pos = _read_record(tokens, pos)
    if header is None:
        raise VfpParseError(f"{kind}: missing header record", file, line)
    items = header.items
    if not items or items[0].is_default:
        raise VfpParseError(f"{kind}: missing table number in header", file, line)
    table_no = items[0].as_int(f"{kind} table number")
    try:
        datum = items[1].as_float("datum depth")
    except VfpParseError:
        raise VfpParseError(
            f"{kind} {table_no}: missing datum depth in header", file, line
        ) from None

    def _enum(idx: int, default: str, valid: tuple[str, ...], what: str) -> str:
        if len(items) <= idx or items[idx].is_default:
            return default
        value = items[idx].as_str(what).upper().strip()
        if value not in valid:
            deck.finding(
                "HEADER_ENUM",
                "ERROR",
                table_no,
                f"{kind} {table_no}: invalid {what} {value!r} "
                f"(valid: {', '.join(valid) or 'blank'})",
                locus={"path": header.file, "line": header.line, "item": idx + 1},
            )
            return value
        return value

    if kind == "VFPPROD":
        flo_type = _enum(2, "GAS", VFPPROD_FLO_TYPES, "FLO type")
        wfr_type = _enum(3, "WCT", WFR_TYPES, "WFR type")
        gfr_type = _enum(4, "GOR", GFR_TYPES, "GFR type")
        thp_type = _enum(5, "THP", THP_TYPES, "THP type")
        alq_type = _enum(6, "", ALQ_TYPES, "ALQ type")
        units = _enum(7, "DEFAULT", UNIT_SYSTEMS, "units")
        tabulated = _enum(8, "BHP", TABULATED_TYPES, "tabulated quantity")
    else:
        # VFPINJ header follows the simulator (OPM/res2df) layout:
        # TABLE DATUM_DEPTH RATE_TYPE PRESSURE_DEF UNITS BODY_DEF
        # (3-item real-world headers default the rest; the ECLIPSE manual's
        # 5-item form without PRESSURE_DEF is NOT what the simulator reads.)
        flo_type = _enum(2, "GAS", VFPINJ_FLO_TYPES, "FLO type")
        _enum(3, "THP", THP_TYPES, "THP type")
        units = _enum(4, "DEFAULT", UNIT_SYSTEMS, "units")
        tabulated = _enum(5, "BHP", ("BHP",), "tabulated quantity")

    # ---- axis vectors ------------------------------------------------------
    axis_order = ("FLO", "THP") if kind == "VFPINJ" else ("FLO", "THP", "WFR", "GFR", "ALQ")
    axis_kinds = {"FLO": flo_type, "THP": thp_type if kind == "VFPPROD" else "THP"}
    if kind == "VFPPROD":
        axis_kinds.update({"WFR": wfr_type, "GFR": gfr_type, "ALQ": alq_type})

    axes: dict[str, VfpAxis] = {}
    for name in axis_order:
        rec, pos = _read_record(tokens, pos)
        if rec is None:
            raise VfpParseError(
                f"{kind} {table_no}: missing {name} axis values", file, line
            )
        if not rec.items:
            raise VfpParseError(
                f"{kind} {table_no}: empty {name} axis vector", rec.file, rec.line
            )
        try:
            values = np.array([it.as_float(f"{name} axis value") for it in rec.items])
        except VfpParseError as e:
            raise VfpParseError(
                f"{kind} {table_no}: {e} (axis {name})", e.path, e.line
            ) from None
        axes[name] = VfpAxis(
            name=name,
            kind=axis_kinds[name],
            values=values,
            unit=resolve_axis_unit(units, name, axis_kinds[name]),
        )

    # VFPINJ degenerates to VFPPROD: WFR/GFR/ALQ axes of length 1
    if kind == "VFPINJ":
        for name, kind_name in (("WFR", "WCT"), ("GFR", "GOR"), ("ALQ", "")):
            axes[name] = VfpAxis(
                name=name,
                kind=kind_name,
                values=np.array([0.0]),
                unit=resolve_axis_unit(units, name, kind_name),
            )

    n_flo = axes["FLO"].values.size
    expected = 1
    for a in ("THP", "WFR", "GFR", "ALQ"):
        expected *= axes[a].values.size
    shape = tuple(axes[a].values.size for a in AXIS_ORDER)
    data = np.full(shape, np.nan)
    seen: set[tuple[int, int, int, int]] = set()

    # ---- data records ------------------------------------------------------
    found = 0
    try:
        while found < expected:
            rec, pos = _read_record(tokens, pos)
            if rec is None:
                raise VfpParseError(
                    f"{kind} {table_no}: expected {expected} data records, found {found}",
                    file,
                    line,
                )
            if not rec.items:
                raise VfpParseError(
                    f"{kind} {table_no}: expected {expected} data records, found {found} "
                    f"(blank record before completion)",
                    rec.file,
                    rec.line,
                )
            idx_names = ("THP", "WFR", "GFR", "ALQ")
            idx_items = rec.items[:4] if kind == "VFPPROD" else rec.items[:1]
            vals_items = rec.items[4:] if kind == "VFPPROD" else rec.items[1:]
            if len(vals_items) != n_flo:
                raise VfpParseError(
                    f"{kind} {table_no}: data record: expected {n_flo} value(s), "
                    f"found {len(vals_items)}",
                    rec.file,
                    rec.line,
                )
            indices: list[int] = []
            for k, it in enumerate(idx_items):
                ax = idx_names[k]
                size = axes[ax].values.size
                v = it.as_int(f"{ax} index")
                if v < 1 or v > size:
                    raise VfpParseError(
                        f"{kind} {table_no}: data record: {ax} index {v} out of range "
                        f"[1..{size}]",
                        rec.file,
                        rec.line,
                    )
                indices.append(v - 1)  # 1-based -> 0-based, once, here
            combo = tuple(indices)
            if combo in seen:
                raise VfpParseError(
                    f"{kind} {table_no}: duplicate data record for indices "
                    f"{tuple(i + 1 for i in combo)}",
                    rec.file,
                    rec.line,
                )
            seen.add(combo)
            vals = np.array([it.as_float(f"{kind} tabulated value") for it in vals_items])
            if not np.all(np.isfinite(vals)):
                bad = int(np.count_nonzero(~np.isfinite(vals)))
                deck.finding(
                    "NON_FINITE",
                    "ERROR",
                    table_no,
                    f"{kind} {table_no}: {bad} non-finite tabulated value(s) in data "
                    f"record {combo}",
                    locus={"path": rec.file, "line": rec.line, "indices": combo},
                )
            if kind == "VFPPROD":
                data[combo[0], combo[1], combo[2], combo[3], :] = vals
            else:
                data[combo[0], 0, 0, 0, :] = vals
            found += 1
    except VfpParseError as e:
        e.table_number = table_no
        raise

    # ---- trailing-record sanity check --------------------------------------
    if pos < len(tokens):
        nxt = tokens[pos]
        if not nxt.is_record_end and (nxt.quoted or _is_numeric(nxt.text)):
            raise VfpParseError(
                f"{kind} {table_no}: expected {expected} data records, found more "
                f"(extra record starting at {nxt.text!r})",
                nxt.file,
                nxt.line,
            )

    try:
        table = VfpTable(
            number=table_no,
            kind=kind,  # type: ignore[arg-type]
            datum_depth=datum,
            unit_system=units,  # type: ignore[arg-type]
            tabulated=tabulated,  # type: ignore[arg-type]
            axes=axes,
            data=data,
            source=SourceRef(path=file, line=line),
        )
    except ValueError as e:
        err = VfpParseError(f"{kind} {table_no}: {e}", file, line)
        err.table_number = table_no
        raise err from None
    return table, pos

def parse_vfp_file(path: str | Path, table_number: int | None = None) -> VfpTable:
    """Parse a standalone include file containing VFP tables.

    Raises VfpParseError with a located message on any structural problem
    (including record-count mismatches); returns the single table otherwise.
    """
    deck = load_deck(path)
    if table_number is None:
        if not deck.tables:
            errs = [f for f in deck.findings if f.check_id == "PARSE"]
            if errs:
                e = errs[0]
                raise VfpParseError(e.message, e.locus.get("path"), e.locus.get("line"))
            raise VfpParseError(f"no VFP tables found in {path}", str(path), 1)
        if len(deck.tables) > 1:
            raise VfpParseError(
                f"{path} contains {len(deck.tables)} VFP tables; pass table_number",
                str(path),
                1,
            )
        table = next(iter(deck.tables.values()))
    else:
        if table_number not in deck.tables:
            errs = [f for f in deck.findings if f.check_id == "PARSE"]
            if errs and errs[0].table_number == table_number:
                e = errs[0]
                raise VfpParseError(e.message, e.locus.get("path"), e.locus.get("line"))
            raise VfpParseError(f"VFP table {table_number} not found in {path}", str(path), 1)
        table = deck.tables[table_number]
    errs = [f for f in deck.findings if f.check_id == "PARSE" and f.table_number == table.number]
    if errs:
        e = errs[0]
        raise VfpParseError(e.message, e.locus.get("path"), e.locus.get("line"))
    return table
