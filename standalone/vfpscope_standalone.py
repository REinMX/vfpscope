#!/usr/bin/env python3
"""VFPScope standalone Streamlit application.

Copy this single file into a Python 3.11+ environment containing NumPy,
Pydantic 2, Plotly, Streamlit, and resdata, then run:

    streamlit run vfpscope_standalone.py -- --deck MODEL.DATA

This standalone edition supports native VFPPROD/VFPINJ parsing, deck consumer
references, QC, curves, heatmaps, table comparison, network visualization, and
simulation-summary operating-envelope coverage from Eclipse/OPM .UNSMRY files.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pydantic import BaseModel, ConfigDict, Field, model_validator

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/core/model.py
# -----------------------------------------------------------------------------

AXIS_ORDER = ('THP', 'WFR', 'GFR', 'ALQ', 'FLO')

VFPPROD_FLO_TYPES = ('OIL', 'LIQ', 'GAS', 'WG', 'TM')

VFPINJ_FLO_TYPES = ('OIL', 'WAT', 'GAS', 'WG', 'TM')

WFR_TYPES = ('WOR', 'WCT', 'WGR', 'WWR', 'WTF')

GFR_TYPES = ('GOR', 'GLR', 'OGR', 'MMW')

THP_TYPES = ('THP',)

ALQ_TYPES = ('GRAT', 'IGLR', 'TGLR', 'PUMP', 'COMP', 'DENO', 'DENG', 'BEAN', '')

UNIT_SYSTEMS = ('METRIC', 'FIELD', 'LAB', 'PVT-M')

TABULATED_TYPES = ('BHP', 'TEMP')

SEVERITY_ORDER = {'INFO': 0, 'WARNING': 1, 'ERROR': 2}

_UNIT_TABLES: dict[str, dict[str, dict[str, str]]] = {'METRIC': {'FLO': {'OIL': 'sm3/day', 'LIQ': 'sm3/day', 'GAS': 'sm3/day', 'WG': 'sm3/day', 'TM': 'kg-M/day', 'WAT': 'sm3/day'}, 'THP': {'THP': 'barsa'}, 'WFR': {'WOR': 'sm3/sm3', 'WCT': 'sm3/sm3', 'WGR': 'sm3/sm3', 'WWR': 'sm3/sm3', 'WTF': ''}, 'GFR': {'GOR': 'sm3/sm3', 'GLR': 'sm3/sm3', 'OGR': 'sm3/sm3', 'MMW': 'kg/kg-M'}, 'ALQ': {'GRAT': 'sm3/day', 'IGLR': 'sm3/sm3', 'TGLR': 'sm3/sm3', 'PUMP': '', 'COMP': '', 'DENO': 'kg/m3', 'DENG': 'kg/m3', 'BEAN': 'mm', '': ''}}, 'FIELD': {'FLO': {'OIL': 'stb/day', 'LIQ': 'stb/day', 'GAS': 'Mscf/day', 'WG': 'lb-M/day', 'TM': 'lb-M/day', 'WAT': 'stb/day'}, 'THP': {'THP': 'psia'}, 'WFR': {'WOR': 'stb/stb', 'WCT': 'stb/stb', 'WGR': 'stb/Mscf', 'WWR': 'stb/Mscf', 'WTF': ''}, 'GFR': {'GOR': 'Mscf/stb', 'GLR': 'Mscf/stb', 'OGR': 'stb/Mscf', 'MMW': 'lb/lb-M'}, 'ALQ': {'GRAT': 'Mscf/day', 'IGLR': 'Mscf/stb', 'TGLR': 'Mscf/stb', 'PUMP': '', 'COMP': '', 'DENO': 'lb/ft3', 'DENG': 'lb/ft3', 'BEAN': '1/64', '': ''}}, 'LAB': {'FLO': {'OIL': 'scc/hr', 'LIQ': 'scc/hr', 'GAS': 'scc/hr', 'WG': 'scc/hr', 'TM': 'lb-M/day', 'WAT': 'scc/hr'}, 'THP': {'THP': 'atma'}, 'WFR': {'WOR': 'scc/scc', 'WCT': 'scc/scc', 'WGR': 'scc/scc', 'WWR': 'scc/scc', 'WTF': ''}, 'GFR': {'GOR': 'scc/scc', 'GLR': 'scc/scc', 'OGR': 'scc/scc', 'MMW': 'lb/lb-M'}, 'ALQ': {'GRAT': 'scc/hr', 'IGLR': 'scc/scc', 'TGLR': 'scc/scc', 'PUMP': '', 'COMP': '', 'DENO': 'gm/cc', 'DENG': 'gm/cc', 'BEAN': 'mm', '': ''}}, 'PVT-M': {'FLO': {'OIL': 'sm3/day', 'LIQ': 'sm3/day', 'GAS': 'sm3/day', 'WG': 'sm3/day', 'TM': 'kg-M/day', 'WAT': 'sm3/day'}, 'THP': {'THP': 'atma'}, 'WFR': {'WOR': 'sm3/sm3', 'WCT': 'sm3/sm3', 'WGR': 'sm3/sm3', 'WWR': 'sm3/sm3', 'WTF': ''}, 'GFR': {'GOR': 'sm3/sm3', 'GLR': 'sm3/sm3', 'OGR': 'sm3/sm3', 'MMW': 'kg/kg-M'}, 'ALQ': {'GRAT': 'sm3/day', 'IGLR': 'sm3/sm3', 'TGLR': 'sm3/sm3', 'PUMP': '', 'COMP': '', 'DENO': 'kg/m3', 'DENG': 'kg/m3', 'BEAN': 'mm', '': ''}}}

_AXIS_UNIT_KEY = {'FLO': 'FLO', 'THP': 'THP', 'WFR': 'WFR', 'GFR': 'GFR', 'ALQ': 'ALQ'}

def resolve_axis_unit(unit_system: str, axis: str, kind: str) -> str:
    """Resolve the display unit for an axis from its unit system and kind."""
    if unit_system not in _UNIT_TABLES:
        return ''
    return _UNIT_TABLES[unit_system].get(_AXIS_UNIT_KEY[axis], {}).get(kind, '')

class SourceRef(BaseModel):
    """File + line provenance for parser errors and display."""
    model_config = ConfigDict(frozen=True)
    path: str
    line: int

    def __str__(self) -> str:
        return f'{self.path}:{self.line}'

class Consumers(BaseModel):
    """Which wells/branches reference a table (role inference result)."""
    model_config = ConfigDict(frozen=True)
    wells: tuple[str, ...] = ()
    branches: tuple[tuple[str, str], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.wells or self.branches)

class VfpAxis(BaseModel):
    """One interpolation axis of a VFP table.

    ``values`` must be 1-D and strictly increasing — validated at the parse
    boundary; a table with a broken axis is reported as a parse finding and
    never built.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    kind: str
    values: np.ndarray
    unit: str = ''

    @model_validator(mode='after')
    def _validate(self) -> VfpAxis:
        v = self.values
        if not isinstance(v, np.ndarray) or v.ndim != 1 or v.size == 0:
            raise ValueError(f'axis {self.name}: values must be a non-empty 1-D array')
        if not np.all(np.isfinite(v)):
            raise ValueError(f'axis {self.name}: values must be finite')
        if not np.all(np.diff(v) > 0):
            raise ValueError(f'axis {self.name}: values must be strictly increasing')
        return self

class Finding(BaseModel):
    """A structured QC finding. Formatting is the caller's job."""
    model_config = ConfigDict(frozen=True)
    check_id: str
    severity: Literal['ERROR', 'WARNING', 'INFO']
    table_number: int
    message: str
    locus: dict = Field(default_factory=dict)
    plot_hint: dict | None = None

class VfpTable(BaseModel):
    """One parsed VFP table (VFPPROD or VFPINJ).

    ``data`` shape: ``(NTHP, NWFR, NGFR, NALQ, NFLO)`` — canonical order
    ``(THP, WFR, GFR, ALQ, FLO)``. Never deviate (see module docstring).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    number: int
    kind: Literal['VFPPROD', 'VFPINJ']
    role: Literal['WELL', 'BRANCH', 'UNKNOWN'] = 'UNKNOWN'
    datum_depth: float
    unit_system: Literal['METRIC', 'FIELD', 'LAB', 'PVT-M', 'DEFAULT'] = 'DEFAULT'
    tabulated: Literal['BHP', 'TEMP'] = 'BHP'
    axes: dict[str, VfpAxis]
    data: np.ndarray
    source: SourceRef
    consumers: Consumers = Field(default_factory=Consumers)

    @model_validator(mode='after')
    def _validate(self) -> VfpTable:
        if set(self.axes) != set(AXIS_ORDER):
            raise ValueError(f'table {self.number}: axes must be exactly {AXIS_ORDER}')
        shape = tuple(len(self.axes[a].values) for a in AXIS_ORDER)
        if self.data.shape != shape:
            raise ValueError(f'table {self.number}: data shape {self.data.shape} != axes shape {shape}')
        if self.kind == 'VFPINJ' and any(self.axes[a].values.size != 1 for a in ('WFR', 'GFR', 'ALQ')):
            raise ValueError(f'table {self.number}: VFPINJ WFR/GFR/ALQ axes must have length 1')
        return self

    @property
    def flo(self) -> VfpAxis:
        return self.axes['FLO']

    @property
    def thp(self) -> VfpAxis:
        return self.axes['THP']

    @property
    def wfr(self) -> VfpAxis:
        return self.axes['WFR']

    @property
    def gfr(self) -> VfpAxis:
        return self.axes['GFR']

    @property
    def alq(self) -> VfpAxis:
        return self.axes['ALQ']

    @property
    def axis_lengths(self) -> dict[str, int]:
        return {a: self.axes[a].values.size for a in AXIS_ORDER}

    @property
    def n_data_records(self) -> int:
        """NTHP x NWFR x NGFR x NALQ — the expected number of data records."""
        n = 1
        for a in ('THP', 'WFR', 'GFR', 'ALQ'):
            n *= self.axes[a].values.size
        return n

    @property
    def label(self) -> str:
        role = {'WELL': 'well', 'BRANCH': 'branch', 'UNKNOWN': 'unassigned'}[self.role]
        return f'{self.kind} #{self.number} ({role})'

    def axis(self, name: str) -> VfpAxis:
        return self.axes[name]

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/core/refs.py
# -----------------------------------------------------------------------------

GAS_PHASES = {'GAS', 'GASWAT', 'GASOIL', 'GASWATOIL', 'OILGAS', 'WATgas'.upper()}

LIQUID_PHASES = {'OIL', 'WAT', 'OILWAT', 'WATOIL'}

def assign_roles(deck) -> None:
    """Infer ``role`` and ``consumers`` for every table in the deck.

    Tables are immutable, so each is replaced by a ``model_copy`` with the
    assigned role/consumers.
    """
    for no in deck.table_order:
        t = deck.tables[no]
        wells = tuple(sorted({w for w, v in deck.well_vfp.items() if v == no}))
        branches = tuple(sorted({(dt, ut) for (dt, ut), v in deck.branch_vfp.items() if v == no}))
        consumers = Consumers(wells=wells, branches=branches)
        if wells and branches:
            role = 'WELL'
        elif wells:
            role = 'WELL'
        elif branches:
            role = 'BRANCH'
        else:
            role = 'UNKNOWN'
        deck.tables[no] = t.model_copy(update={'role': role, 'consumers': consumers})
    known = set(deck.tables)
    for well, no in deck.well_vfp.items():
        if no not in known:
            deck.finding('MISSING_TABLE', 'ERROR', no, f'well {well} references VFP table {no}, which is not defined in the deck', locus={'well': well})
    for (dt, ut), no in deck.branch_vfp.items():
        if no not in known:
            deck.finding('MISSING_TABLE', 'ERROR', no, f'branch {dt}->{ut} references VFP table {no}, which is not defined in the deck', locus={'branch': [dt, ut]})

def referencing_wells(deck, table_no: int) -> tuple[str, ...]:
    return tuple(sorted({w for w, v in deck.well_vfp.items() if v == table_no}))

def referencing_branches(deck, table_no: int) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({(dt, ut) for (dt, ut), v in deck.branch_vfp.items() if v == table_no}))

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/core/parse/native.py
# -----------------------------------------------------------------------------

_NUM_RE = re.compile('^[+-]?(\\d+(\\.\\d*)?|\\.\\d+)([EeDd][+-]?\\d+)?$')

_REPEAT_RE = re.compile('^(\\d+)\\*(.*)$')

_MAX_INCLUDE_DEPTH = 50

class VfpParseError(Exception):
    """Located parse error. ``str(e)`` carries file:line; ``e.raw`` is bare."""

    def __init__(self, message: str, path: str | None=None, line: int | None=None):
        self.path = path
        self.line = line
        self.raw = message
        loc = f'{path}:{line}: ' if path and line else ''
        super().__init__(f'{loc}{message}')

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
    i, n = (0, len(text))
    line = 1
    while i < n:
        c = text[i]
        if c in ' \t\r':
            i += 1
            continue
        if c == '\n':
            line += 1
            i += 1
            continue
        if c == '-' and text[i:i + 2] == '--':
            while i < n and text[i] != '\n':
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
                raise VfpParseError('unterminated quoted string', path, line)
            toks.append(Token(''.join(buf), path, line, quoted=True))
            i = j + 1
            continue
        if c == '/':
            toks.append(Token('/', path, line, is_record_end=True))
            i += 1
            continue
        j = i
        while j < n:
            ch = text[j]
            if ch in " \t\r\n/'" or (ch == '-' and text[j:j + 2] == '--'):
                break
            j += 1
        if j < n and text[j] == "'" and _REPEAT_RE.match(text[i:j]):
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
                raise VfpParseError('unterminated quoted string', path, line)
            toks.append(Token(''.join(buf), path, line))
            i = k + 1
            continue
        toks.append(Token(text[i:j], path, line))
        i = j
    return toks

def parse_string(text: str, path: str='<string>') -> list[Token]:
    return tokenize(text, path)

def _is_numeric(text: str) -> bool:
    return bool(_NUM_RE.match(text))

def _as_float(text: str) -> float | None:
    """Parse a numeric token; None if it is not numeric."""
    if _is_numeric(text):
        return float(text.replace('D', 'E').replace('d', 'e'))
    return None

@dataclass(frozen=True)
class Item:
    value: float | str | None
    file: str
    line: int

    @property
    def is_default(self) -> bool:
        return self.value is None

    def as_int(self, what: str) -> int:
        if self.value is None:
            raise VfpParseError(f'missing (default) value for {what}', self.file, self.line)
        if not isinstance(self.value, float) or not float(self.value).is_integer():
            raise VfpParseError(f'{what}: expected an integer, got {self.value!r}', self.file, self.line)
        return int(self.value)

    def as_float(self, what: str) -> float:
        if self.value is None:
            raise VfpParseError(f'missing (default) value for {what}', self.file, self.line)
        if not isinstance(self.value, float):
            raise VfpParseError(f'{what}: expected a number, got {self.value!r}', self.file, self.line)
        return self.value

    def as_str(self, what: str) -> str:
        if self.value is None:
            raise VfpParseError(f'missing (default) value for {what}', self.file, self.line)
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
        if text == '*':
            items.append(Item(None, tok.file, tok.line))
            continue
        m = _REPEAT_RE.match(text)
        if m:
            count = int(m.group(1))
            rest = m.group(2)
            if not rest:
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
        return (None, pos)
    start = pos
    while pos < len(tokens) and (not tokens[pos].is_record_end):
        pos += 1
    if pos >= len(tokens):
        return (None, start)
    rec_tokens = tokens[start:pos]
    items = expand_record_items(rec_tokens)
    file = rec_tokens[0].file if rec_tokens else tokens[start].file
    line = rec_tokens[0].line if rec_tokens else tokens[start].line
    return (Record(tuple(items), file, line), pos + 1)

@dataclass
class Deck:
    """Everything extracted from one deck/include tree."""
    path: str
    unit_system: str = 'DEFAULT'
    tables: dict[int, VfpTable] = field(default_factory=dict)
    table_order: list[int] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    well_vfp: dict[str, int] = field(default_factory=dict)
    branch_vfp: dict[tuple[str, str], int] = field(default_factory=dict)
    branches: list[tuple[str, str, int]] = field(default_factory=list)
    nodes: list[tuple[str, float | None]] = field(default_factory=list)
    wells_datum: dict[str, float] = field(default_factory=dict)
    well_phases: dict[str, str] = field(default_factory=dict)
    gaslift_wells: set[str] = field(default_factory=set)

    def finding(self, check_id: str, severity: str, table_number: int, message: str, locus: dict | None=None, plot_hint: dict | None=None) -> None:
        self.findings.append(Finding(check_id=check_id, severity=severity, table_number=table_number, message=message, locus=locus or {}, plot_hint=plot_hint))

_INTEREST_KEYWORDS = {'VFPPROD', 'VFPINJ', 'INCLUDE', 'PATHS', 'WCONPROD', 'WCONINJE', 'BRANPROP', 'NODEPROP', 'WELSPECS', 'WLIFTOPT', 'METRIC', 'FIELD', 'LAB', 'PVT-M'}

_UNIT_KEYWORDS = {'METRIC': 'METRIC', 'FIELD': 'FIELD', 'LAB': 'LAB', 'PVT-M': 'PVT-M'}

def _find_keyword_start(tokens: list[Token], pos: int) -> int:
    """Next token that is a bare (unquoted, non-numeric) identifier."""
    while pos < len(tokens):
        t = tokens[pos]
        if not t.is_record_end and (not t.quoted) and (not _is_numeric(t.text)):
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
    tokens = tokenize(Path(root).read_text(encoding='utf-8', errors='replace'), root)
    aliases: dict[str, str] = {}
    _scan_tokens(deck, tokens, 0, [root], aliases)
    assign_roles(deck)
    return deck

def _scan_tokens(deck: Deck, tokens: list[Token], pos: int, include_stack: list[str], aliases: dict[str, str]) -> None:
    """Scan a token list for keywords; INCLUDEs recurse (stack = recursion)."""
    while True:
        pos = _find_keyword_start(tokens, pos)
        if pos >= len(tokens):
            return
        kw = tokens[pos].text.upper()
        pos += 1
        if kw == 'VFPPROD':
            _parse_vfp_keyword(deck, 'VFPPROD', tokens, pos)
            pos = _find_keyword_start(tokens, pos)
            continue
        if kw == 'VFPINJ':
            _parse_vfp_keyword(deck, 'VFPINJ', tokens, pos)
            pos = _find_keyword_start(tokens, pos)
            continue
        if kw == 'INCLUDE':
            rec, pos = _read_record(tokens, pos)
            if rec is None:
                raise VfpParseError('INCLUDE without a record', tokens[pos - 1].file, tokens[pos - 1].line)
            for item in rec.items:
                name = item.as_str('INCLUDE filename')
                resolved = _resolve_include(name, item.file, aliases)
                abs_path = str(Path(resolved).resolve())
                if abs_path in include_stack:
                    raise VfpParseError(f'include cycle detected: {abs_path}', item.file, item.line)
                if len(include_stack) >= _MAX_INCLUDE_DEPTH:
                    raise VfpParseError('include depth limit exceeded', item.file, item.line)
                if not Path(abs_path).exists():
                    raise VfpParseError(f'include file not found: {name} (resolved {abs_path})', item.file, item.line)
                sub = tokenize(Path(abs_path).read_text(encoding='utf-8', errors='replace'), abs_path)
                _scan_tokens(deck, sub, 0, include_stack + [abs_path], aliases)
            continue
        if kw == 'PATHS':
            while True:
                rec, pos = _read_record(tokens, pos)
                if rec is None or not rec.items:
                    if rec is None:
                        raise VfpParseError("unterminated PATHS block (missing blank '/' record)", tokens[pos - 1].file, tokens[pos - 1].line)
                    break
                for i in range(0, len(rec.items) - 1, 2):
                    name = rec.items[i].as_str('PATHS alias name')
                    target = rec.items[i + 1].as_str('PATHS alias path')
                    aliases[name.upper()] = target
            continue
        if kw in _UNIT_KEYWORDS:
            deck.unit_system = kw
            continue
        if kw in ('WCONPROD', 'WCONINJE', 'BRANPROP', 'NODEPROP', 'WELSPECS', 'WLIFTOPT'):
            pos = _consume_ref_keyword(deck, kw, tokens, pos)
            continue

def _resolve_include(name: str, file: str, aliases: dict[str, str]) -> str:
    base = Path(file).parent
    if name.startswith('$'):
        var, _, rest = name[1:].partition('/')
        if var.upper() not in aliases:
            raise VfpParseError(f"unknown PATHS alias '${var}' in include {name!r}", file, 0)
        return str(Path(aliases[var.upper()]) / rest)
    return str(base / name)

def _consume_ref_keyword(deck: Deck, kw: str, tokens: list[Token], pos: int) -> int:
    """Collect records for a schedule/reference keyword until the blank record."""
    n_records = 0
    while True:
        rec, pos = _read_record(tokens, pos)
        if rec is None:
            raise VfpParseError(f"unterminated {kw} block (missing blank '/' record)", tokens[pos - 1].file if pos else deck.path, tokens[pos - 1].line if pos else 0)
        if not rec.items:
            return pos
        n_records += 1
        try:
            if kw == 'WCONPROD':
                well = rec.items[0].as_str('well name').upper()
                if len(rec.items) > 10 and (not rec.items[10].is_default):
                    vfp = rec.items[10].as_int(f'WCONPROD VFP table for {well}')
                    if vfp > 0:
                        deck.well_vfp[well] = vfp
            elif kw == 'WCONINJE':
                well = rec.items[0].as_str('well name').upper()
                if len(rec.items) > 8 and (not rec.items[8].is_default):
                    vfp = rec.items[8].as_int(f'WCONINJE VFP table for {well}')
                    if vfp > 0:
                        deck.well_vfp[well] = vfp
            elif kw == 'BRANPROP':
                dt = rec.items[0].as_str('branch downtree node').upper()
                ut = rec.items[1].as_str('branch uptree node').upper()
                vfp = rec.items[2].as_int(f'BRANPROP VFP table for {dt}-{ut}')
                if vfp > 0:
                    deck.branch_vfp[dt, ut] = vfp
                    deck.branches.append((dt, ut, vfp))
            elif kw == 'NODEPROP':
                name = rec.items[0].as_str('node name').upper()
                pressure = rec.items[1].as_float(f'NODEPROP pressure for {name}') if len(rec.items) > 1 and (not rec.items[1].is_default) else None
                deck.nodes.append((name, pressure))
            elif kw == 'WELSPECS':
                well = rec.items[0].as_str('well name').upper()
                if len(rec.items) > 4 and (not rec.items[4].is_default):
                    deck.wells_datum[well] = rec.items[4].as_float(f'datum for {well}')
                if len(rec.items) > 5 and (not rec.items[5].is_default):
                    deck.well_phases[well] = rec.items[5].as_str(f'phase for {well}').upper()
            elif kw == 'WLIFTOPT':
                deck.gaslift_wells.add(rec.items[0].as_str('well name').upper())
        except VfpParseError as e:
            deck.finding('PARSE', 'ERROR', 0, f'in {kw}: {e}', locus={'path': e.path, 'line': e.line})
    return pos

def _parse_vfp_keyword(deck: Deck, kind: str, tokens: list[Token], pos: int) -> None:
    try:
        table, pos = _parse_table(deck, kind, tokens, pos)
    except VfpParseError as e:
        deck.finding('PARSE', 'ERROR', getattr(e, 'table_number', 0) or 0, getattr(e, 'raw', str(e)), locus={'path': e.path, 'line': e.line})
        _skip_to_next_keyword(tokens, pos)
        return
    if table.number in deck.tables:
        deck.finding('DUP_TABLE', 'ERROR', table.number, f'duplicate VFP table number {table.number} in deck (first occurrence at {deck.tables[table.number].source})', locus={'path': table.source.path, 'line': table.source.line})
        return
    deck.tables[table.number] = table
    deck.table_order.append(table.number)

def _skip_to_next_keyword(tokens: list[Token], pos: int) -> None:
    _find_keyword_start(tokens, pos)

def _parse_table(deck: Deck, kind: str, tokens: list[Token], pos: int) -> tuple[VfpTable, int]:
    """Parse one VFPPROD/VFPINJ keyword block; returns (table, next_pos)."""
    kw_token = tokens[pos - 1]
    file, line = (kw_token.file, kw_token.line)
    header, pos = _read_record(tokens, pos)
    if header is None:
        raise VfpParseError(f'{kind}: missing header record', file, line)
    items = header.items
    if not items or items[0].is_default:
        raise VfpParseError(f'{kind}: missing table number in header', file, line)
    table_no = items[0].as_int(f'{kind} table number')
    try:
        datum = items[1].as_float('datum depth')
    except VfpParseError:
        raise VfpParseError(f'{kind} {table_no}: missing datum depth in header', file, line) from None

    def _enum(idx: int, default: str, valid: tuple[str, ...], what: str) -> str:
        if len(items) <= idx or items[idx].is_default:
            return default
        value = items[idx].as_str(what).upper().strip()
        if value not in valid:
            deck.finding('HEADER_ENUM', 'ERROR', table_no, f"{kind} {table_no}: invalid {what} {value!r} (valid: {', '.join(valid) or 'blank'})", locus={'path': header.file, 'line': header.line, 'item': idx + 1})
            return value
        return value
    if kind == 'VFPPROD':
        flo_type = _enum(2, 'GAS', VFPPROD_FLO_TYPES, 'FLO type')
        wfr_type = _enum(3, 'WCT', WFR_TYPES, 'WFR type')
        gfr_type = _enum(4, 'GOR', GFR_TYPES, 'GFR type')
        thp_type = _enum(5, 'THP', THP_TYPES, 'THP type')
        alq_type = _enum(6, '', ALQ_TYPES, 'ALQ type')
        units = _enum(7, 'DEFAULT', UNIT_SYSTEMS, 'units')
        tabulated = _enum(8, 'BHP', TABULATED_TYPES, 'tabulated quantity')
    else:
        flo_type = _enum(2, 'GAS', VFPINJ_FLO_TYPES, 'FLO type')
        _enum(3, 'THP', THP_TYPES, 'THP type')
        units = _enum(4, 'DEFAULT', UNIT_SYSTEMS, 'units')
        tabulated = _enum(5, 'BHP', ('BHP',), 'tabulated quantity')
    axis_order = ('FLO', 'THP') if kind == 'VFPINJ' else ('FLO', 'THP', 'WFR', 'GFR', 'ALQ')
    axis_kinds = {'FLO': flo_type, 'THP': thp_type if kind == 'VFPPROD' else 'THP'}
    if kind == 'VFPPROD':
        axis_kinds.update({'WFR': wfr_type, 'GFR': gfr_type, 'ALQ': alq_type})
    axes: dict[str, VfpAxis] = {}
    for name in axis_order:
        rec, pos = _read_record(tokens, pos)
        if rec is None:
            raise VfpParseError(f'{kind} {table_no}: missing {name} axis values', file, line)
        if not rec.items:
            raise VfpParseError(f'{kind} {table_no}: empty {name} axis vector', rec.file, rec.line)
        try:
            values = np.array([it.as_float(f'{name} axis value') for it in rec.items])
        except VfpParseError as e:
            raise VfpParseError(f'{kind} {table_no}: {e} (axis {name})', e.path, e.line) from None
        axes[name] = VfpAxis(name=name, kind=axis_kinds[name], values=values, unit=resolve_axis_unit(units, name, axis_kinds[name]))
    if kind == 'VFPINJ':
        for name, kind_name in (('WFR', 'WCT'), ('GFR', 'GOR'), ('ALQ', '')):
            axes[name] = VfpAxis(name=name, kind=kind_name, values=np.array([0.0]), unit=resolve_axis_unit(units, name, kind_name))
    n_flo = axes['FLO'].values.size
    expected = 1
    for a in ('THP', 'WFR', 'GFR', 'ALQ'):
        expected *= axes[a].values.size
    shape = tuple(axes[a].values.size for a in AXIS_ORDER)
    data = np.full(shape, np.nan)
    seen: set[tuple[int, int, int, int]] = set()
    found = 0
    try:
        while found < expected:
            rec, pos = _read_record(tokens, pos)
            if rec is None:
                raise VfpParseError(f'{kind} {table_no}: expected {expected} data records, found {found}', file, line)
            if not rec.items:
                raise VfpParseError(f'{kind} {table_no}: expected {expected} data records, found {found} (blank record before completion)', rec.file, rec.line)
            idx_names = ('THP', 'WFR', 'GFR', 'ALQ')
            idx_items = rec.items[:4] if kind == 'VFPPROD' else rec.items[:1]
            vals_items = rec.items[4:] if kind == 'VFPPROD' else rec.items[1:]
            if len(vals_items) != n_flo:
                raise VfpParseError(f'{kind} {table_no}: data record: expected {n_flo} value(s), found {len(vals_items)}', rec.file, rec.line)
            indices: list[int] = []
            for k, it in enumerate(idx_items):
                ax = idx_names[k]
                size = axes[ax].values.size
                v = it.as_int(f'{ax} index')
                if v < 1 or v > size:
                    raise VfpParseError(f'{kind} {table_no}: data record: {ax} index {v} out of range [1..{size}]', rec.file, rec.line)
                indices.append(v - 1)
            combo = tuple(indices)
            if combo in seen:
                raise VfpParseError(f'{kind} {table_no}: duplicate data record for indices {tuple(i + 1 for i in combo)}', rec.file, rec.line)
            seen.add(combo)
            vals = np.array([it.as_float(f'{kind} tabulated value') for it in vals_items])
            if not np.all(np.isfinite(vals)):
                bad = int(np.count_nonzero(~np.isfinite(vals)))
                deck.finding('NON_FINITE', 'ERROR', table_no, f'{kind} {table_no}: {bad} non-finite tabulated value(s) in data record {combo}', locus={'path': rec.file, 'line': rec.line, 'indices': combo})
            if kind == 'VFPPROD':
                data[combo[0], combo[1], combo[2], combo[3], :] = vals
            else:
                data[combo[0], 0, 0, 0, :] = vals
            found += 1
    except VfpParseError as e:
        e.table_number = table_no
        raise
    if pos < len(tokens):
        nxt = tokens[pos]
        if not nxt.is_record_end and (nxt.quoted or _is_numeric(nxt.text)):
            raise VfpParseError(f'{kind} {table_no}: expected {expected} data records, found more (extra record starting at {nxt.text!r})', nxt.file, nxt.line)
    try:
        table = VfpTable(number=table_no, kind=kind, datum_depth=datum, unit_system=units, tabulated=tabulated, axes=axes, data=data, source=SourceRef(path=file, line=line))
    except ValueError as e:
        err = VfpParseError(f'{kind} {table_no}: {e}', file, line)
        err.table_number = table_no
        raise err from None
    return (table, pos)

def parse_vfp_file(path: str | Path, table_number: int | None=None) -> VfpTable:
    """Parse a standalone include file containing VFP tables.

    Raises VfpParseError with a located message on any structural problem
    (including record-count mismatches); returns the single table otherwise.
    """
    deck = load_deck(path)
    if table_number is None:
        if not deck.tables:
            errs = [f for f in deck.findings if f.check_id == 'PARSE']
            if errs:
                e = errs[0]
                raise VfpParseError(e.message, e.locus.get('path'), e.locus.get('line'))
            raise VfpParseError(f'no VFP tables found in {path}', str(path), 1)
        if len(deck.tables) > 1:
            raise VfpParseError(f'{path} contains {len(deck.tables)} VFP tables; pass table_number', str(path), 1)
        table = next(iter(deck.tables.values()))
    else:
        if table_number not in deck.tables:
            errs = [f for f in deck.findings if f.check_id == 'PARSE']
            if errs and errs[0].table_number == table_number:
                e = errs[0]
                raise VfpParseError(e.message, e.locus.get('path'), e.locus.get('line'))
            raise VfpParseError(f'VFP table {table_number} not found in {path}', str(path), 1)
        table = deck.tables[table_number]
    errs = [f for f in deck.findings if f.check_id == 'PARSE' and f.table_number == table.number]
    if errs:
        e = errs[0]
        raise VfpParseError(e.message, e.locus.get('path'), e.locus.get('line'))
    return table

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/core/derive.py
# -----------------------------------------------------------------------------

def dp_curve(table: VfpTable, thp_index: int=0, wfr: int=0, gfr: int=0, alq: int=0) -> tuple[np.ndarray, np.ndarray]:
    """(FLO values, BHP - THP) for one slice — the pressure drop across the
    tubing (well role) or across the branch (branch role)."""
    flo = table.axes['FLO'].values
    thp = table.axes['THP'].values[thp_index]
    dp = table.data[thp_index, wfr, gfr, alq, :] - thp
    return (flo, dp)

def gradient(table: VfpTable, thp_index: int=0, wfr: int=0, gfr: int=0, alq: int=0) -> tuple[np.ndarray, np.ndarray]:
    """d(BHP)/dFLO at midpoints for one slice."""
    flo, dp = dp_curve(table, thp_index, wfr, gfr, alq)
    mid = 0.5 * (flo[:-1] + flo[1:])
    grad = np.diff(dp) / np.diff(flo)
    return (mid, grad)

def turning_points(table: VfpTable) -> list[dict]:
    """Turning points (unstable limb) per curve: {thp,wfr,gfr,alq,flo,value}."""
    flo = table.axes['FLO'].values
    out: list[dict] = []
    for ti in range(table.axes['THP'].values.size):
        for w in range(table.axes['WFR'].values.size):
            for g in range(table.axes['GFR'].values.size):
                for a in range(table.axes['ALQ'].values.size):
                    curve = table.data[ti, w, g, a, :]
                    d = np.sign(np.diff(curve))
                    turns = np.where(np.diff(d) != 0)[0]
                    for k in turns:
                        k = int(k) + 1
                        out.append({'thp': ti, 'wfr': w, 'gfr': g, 'alq': a, 'flo': float(flo[k]), 'value': float(curve[k])})
    return out

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/core/interp.py
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LookupResult:
    value: float | np.ndarray
    clamped: dict[str, str | None] = None
    clamped_masks: dict[str, tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def any_clamped(self) -> bool:
        if self.clamped is not None:
            return any(v is not None for v in self.clamped.values())
        if self.clamped_masks:
            return any((bool(np.any(lo) or np.any(hi)) for lo, hi in self.clamped_masks.values()))
        return False

def _axis_interp(values: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Bracketing indices, interpolation weights, clamp flags for one axis."""
    v = values
    low = x < v[0]
    high = x > v[-1]
    xc = np.clip(x, v[0], v[-1])
    if v.size == 1:
        z = np.zeros(x.shape, dtype=int)
        return (z, z, np.zeros(x.shape), (low, high))
    i0 = np.searchsorted(v, xc, side='right') - 1
    i0 = np.clip(i0, 0, v.size - 2)
    i1 = i0 + 1
    denom = v[i1] - v[i0]
    t = np.zeros_like(xc)
    np.divide(xc - v[i0], denom, out=t, where=denom != 0)
    return (i0, i1, t, (low, high))

def _lookup_many(table: VfpTable, coords: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    n = len(next(iter(coords.values())))
    i0s: dict[str, np.ndarray] = {}
    i1s: dict[str, np.ndarray] = {}
    ts: dict[str, np.ndarray] = {}
    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for a in AXIS_ORDER:
        x = coords[a]
        i0, i1, t, m = _axis_interp(table.axes[a].values, x)
        i0s[a] = i0
        i1s[a] = i1
        ts[a] = t
        masks[a] = m
    value = np.zeros(n)
    for corner in product((0, 1), repeat=len(AXIS_ORDER)):
        w = np.ones(n)
        idx: list[np.ndarray] = []
        for a, c in zip(AXIS_ORDER, corner, strict=True):
            idx.append(i1s[a] if c else i0s[a])
            w = w * (ts[a] if c else 1.0 - ts[a])
        value += w * table.data[tuple(idx)]
    return (value, masks)

def lookup(table: VfpTable, *, flo: float | np.ndarray, thp: float | np.ndarray, wfr: float | np.ndarray | None=None, gfr: float | np.ndarray | None=None, alq: float | np.ndarray | None=None) -> LookupResult:
    """Evaluate the table at a query point (scalar or equal-length arrays).

    Missing ratio/ALQ coordinates default to the **first** value of that axis
    (e.g. WCT=0, lowest GOR), which is the natural base case.
    """
    defaults = {a: table.axes[a].values[0] for a in ('WFR', 'GFR', 'ALQ')}
    given = {'WFR': wfr, 'GFR': gfr, 'ALQ': alq}
    coords: dict[str, np.ndarray] = {'FLO': np.asarray(flo, dtype=float), 'THP': np.asarray(thp, dtype=float)}
    for a in ('WFR', 'GFR', 'ALQ'):
        v = given[a]
        coords[a] = np.asarray(defaults[a] if v is None else v, dtype=float)
    scalar = all(c.ndim == 0 for c in coords.values())
    if scalar:
        coords = {a: c.reshape(1) for a, c in coords.items()}
    else:
        n = coords['FLO'].size
        for _axis, coordinate in coords.items():
            if coordinate.ndim > 0 and coordinate.size != n:
                raise ValueError('all query coordinates must be scalars or equal-length arrays')
        coords = {a: c if c.ndim > 0 else np.full(n, float(c)) for a, c in coords.items()}
    values, masks = _lookup_many(table, coords)
    if scalar:
        clamped = {a: 'low' if m[0][0] else 'high' if m[1][0] else None for a, m in masks.items()}
        return LookupResult(value=float(values[0]), clamped=clamped)
    return LookupResult(value=values, clamped_masks=masks)

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/core/coverage.py
# -----------------------------------------------------------------------------

_VECTORS: dict[str, dict[str, str | tuple[str, ...] | None]] = {'FLO': {'OIL': 'WOPR', 'LIQ': ('WOPR', 'WWPR'), 'GAS': 'WGPR', 'WG': ('WOPR', 'WWPR', 'WGPR'), 'TM': ('WOPR', 'WWPR', 'WGPR'), 'WAT': 'WWIR'}, 'THP': {'THP': 'WTHP'}, 'WFR': {'WOR': ('WWPR', 'WOPR'), 'WCT': 'WWCT', 'WGR': 'WWGR', 'WWR': None, 'WTF': None}, 'GFR': {'GOR': 'WGOR', 'GLR': 'WGLR', 'OGR': None, 'MMW': None}, 'ALQ': {'GRAT': 'WGLIR', 'IGLR': None, 'TGLR': None, 'PUMP': None, 'COMP': None, 'DENO': None, 'DENG': None, 'BEAN': None, '': None}}

class CoverageUnavailable(RuntimeError):
    """resdata (or the summary file) is not available."""

@dataclass
class CoverageReport:
    """Per-well coverage of a table by one run's operating envelope."""
    table_number: int
    well: str
    n_timesteps: int
    axis_values: dict[str, np.ndarray | None] = field(default_factory=dict)
    clamped_low: dict[str, np.ndarray | None] = field(default_factory=dict)
    clamped_high: dict[str, np.ndarray | None] = field(default_factory=dict)
    bhp: np.ndarray | None = None

    @property
    def axes_with_data(self) -> list[str]:
        return [a for a in AXIS_ORDER if self.axis_values.get(a) is not None]

    def fraction_clamped(self, axis: str) -> float:
        lo = self.clamped_low.get(axis)
        hi = self.clamped_high.get(axis)
        if lo is None or hi is None:
            return 0.0
        return float(np.mean(lo | hi))

    def fraction_clamped_low(self, axis: str) -> float:
        lo = self.clamped_low.get(axis)
        return float(np.mean(lo)) if lo is not None else 0.0

    def fraction_clamped_high(self, axis: str) -> float:
        hi = self.clamped_high.get(axis)
        return float(np.mean(hi)) if hi is not None else 0.0

    def run_max_over_table(self, axis: str, table: VfpTable) -> float | None:
        v = self.axis_values.get(axis)
        tmax = float(table.axes[axis].values[-1])
        if v is None or tmax <= 0:
            return None
        return float(v.max()) / tmax

    def run_min_under_table(self, axis: str, table: VfpTable) -> float | None:
        v = self.axis_values.get(axis)
        tmin = float(table.axes[axis].values[0])
        if v is None or tmin <= 0:
            return None
        return float(v.min()) / tmin

def _resolve(smry, well: str, spec) -> np.ndarray | None:
    if spec is None:
        return None
    names = spec if isinstance(spec, tuple) else (spec,)
    try:
        parts = []
        for n in names:
            key = f'{n}:{well}'
            try:
                vec = np.asarray(smry.numpy_vector(key), dtype=float)
            except (KeyError, TypeError, ValueError):
                return None
            parts.append(vec)
        if not parts:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out
    except Exception:
        return None

def _ratio_from_components(parts: list[np.ndarray]) -> np.ndarray:
    """WOR-style ratio: num/den with den==0 guarded to 0."""
    num, den = (parts[0], parts[1])
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out

def coverage_from_summary(table: VfpTable, well: str, smry) -> CoverageReport:
    """Evaluate one well's run envelope against one table."""
    n = None
    axis_values: dict[str, np.ndarray | None] = {}
    for a in AXIS_ORDER:
        kind = table.axes[a].kind
        spec = _VECTORS[a].get(kind)
        if a == 'WFR' and kind == 'WOR':
            num = _resolve(smry, well, 'WWPR')
            den = _resolve(smry, well, 'WOPR')
            vec = _ratio_from_components([num, den]) if num is not None and den is not None else None
        else:
            vec = _resolve(smry, well, spec)
        axis_values[a] = vec
        if vec is not None:
            n = vec.size
    if n is None:
        raise CoverageUnavailable(f'no summary vectors match table {table.number} for well {well}')
    wbhp = _resolve(smry, well, 'WBHP')
    coords = {}
    for a in AXIS_ORDER:
        v = axis_values[a]
        coords[a] = v if v is not None else np.full(n, float(table.axes[a].values[0]))
    _, masks = _lookup_many(table, coords)
    clamped_low = {a: masks[a][0] if axis_values[a] is not None else None for a in AXIS_ORDER}
    clamped_high = {a: masks[a][1] if axis_values[a] is not None else None for a in AXIS_ORDER}
    return CoverageReport(table_number=table.number, well=well, n_timesteps=n, axis_values=axis_values, clamped_low=clamped_low, clamped_high=clamped_high, bhp=wbhp)

def load_summary(path: str | Path):
    """Load a summary file via resdata (case stem without extension)."""
    p = Path(path)
    stem = str(p.with_suffix(''))
    try:
        from resdata.summary import Summary
    except ImportError as e:
        raise CoverageUnavailable('resdata is not installed in this Python environment') from e
    return Summary(stem)

def coverage_for_deck(deck, smry_path: str | Path) -> dict[int, list[CoverageReport]]:
    """Coverage reports for every (table, consuming well) pair in a deck."""
    smry = load_summary(smry_path)
    out: dict[int, list[CoverageReport]] = {}
    for no in deck.table_order:
        t = deck.tables[no]
        reports = []
        for well in t.consumers.wells:
            try:
                reports.append(coverage_from_summary(t, well, smry))
            except CoverageUnavailable:
                continue
        if reports:
            out[no] = reports
    return out

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/core/qc/checks.py
# -----------------------------------------------------------------------------

@dataclass
class QCContext:
    """Deck-level context needed by consistency/coverage checks."""
    deck: object | None = None
    thresholds: dict[str, float] = field(default_factory=lambda: {'datum_tolerance_m': 100.0, 'clamp_warning_frac': 0.05, 'absurd_bhp_bar': 10000.0})
    coverage: dict[int, object] | None = None

def _f(table: VfpTable, severity: str, check_id: str, message: str, locus=None, plot_hint=None) -> Finding:
    return Finding(check_id=check_id, severity=severity, table_number=table.number, message=message, locus=locus or {}, plot_hint=plot_hint)

def check_record_count(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Data record count must be NTHP x NWFR x NGFR x NALQ.

    Enforced by the native parser (PARSE finding); re-validated here for
    programmatically built tables.
    """
    if table.data.size != table.n_data_records * table.axis_lengths['FLO']:
        return [_f(table, 'ERROR', 'RECORD_COUNT', f"data contains {table.data.size // table.axis_lengths['FLO']} records, expected {table.n_data_records}")]
    return []

def check_record_width(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Every data record must carry exactly NFLO values (parse-enforced)."""
    return []

def check_axes_monotonic(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Axis vectors strictly increasing, no duplicates (parse-enforced)."""
    return []

def check_non_finite(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """NaN / inf entries in the hypercube."""
    bad = ~np.isfinite(table.data)
    if not bad.any():
        return []
    idx = np.argwhere(bad)[0]
    return [_f(table, 'ERROR', 'NON_FINITE', f'{int(bad.sum())} non-finite value(s) in the hypercube (first at {dict(zip(AXIS_ORDER, idx.tolist(), strict=True))})', locus={'indices': idx.tolist()})]

def check_indices_in_range(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Data-record indices within axis bounds (parse-enforced)."""
    return []

def check_header_enums(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Header type strings must be in the valid enumerations (parse-enforced)."""
    return []

def check_duplicate_tables(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Duplicate table numbers within the deck (deck-level, parse-enforced)."""
    return []

def check_thp_monotonic(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """BHP must increase with THP at fixed (WFR, GFR, ALQ, FLO)."""
    thp = table.axes['THP'].values
    if thp.size < 2:
        return []
    d = np.diff(table.data, axis=0)
    bad = d <= -1e-09
    if not bad.any():
        return []
    n_slices = int(bad.any(axis=0).sum())
    first = np.argwhere(bad)[0]
    slice_idx = first[1:].tolist()
    thp_lo = thp[first[0]]
    thp_hi = thp[first[0] + 1]
    return [_f(table, 'ERROR', 'THP_MONOTONIC', f"BHP decreases with THP in {n_slices} slice(s) (first: {table.axes['FLO'].kind}={table.axes['FLO'].values[first[4]]:g}, THP {thp_lo:g}->{thp_hi:g}, WFR/GFR/ALQ idx {slice_idx[1:]})", locus={'slice': slice_idx, 'thp': [float(thp_lo), float(thp_hi)]}, plot_hint={'kind': 'thp_monotonic', 'slice': slice_idx})]

def check_crossing_thp_curves(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """THP curves (BHP vs FLO) must not cross at fixed (WFR, GFR, ALQ)."""
    flo = table.axes['FLO'].values
    n_thp = table.axes['THP'].values.size
    if n_thp < 2 or flo.size < 2:
        return []
    findings: list[Finding] = []
    for w in range(table.axes['WFR'].values.size):
        for g in range(table.axes['GFR'].values.size):
            for a in range(table.axes['ALQ'].values.size):
                slab = table.data[:, w, g, a, :]
                for i in range(n_thp):
                    for j in range(i + 1, n_thp):
                        d = slab[i] - slab[j]
                        signs = np.sign(d)
                        cross = np.where(np.diff(signs) != 0)[0]
                        if cross.size:
                            k = int(cross[0])
                            findings.append(_f(table, 'ERROR', 'CROSSING', f"THP curves {i + 1} and {j + 1} cross near {table.axes['FLO'].kind}={flo[k]:g} (WFR/GFR/ALQ idx {w},{g},{a})", locus={'wfr': w, 'gfr': g, 'alq': a, 'flo': int(k), 'thp_pair': [i, j]}, plot_hint={'kind': 'crossing', 'x': [float(flo[k])], 'y': [float(slab[i][k])]}))
                            if len(findings) >= 20:
                                return findings
    return findings

def check_unstable_branch(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Non-monotonic BHP vs FLO: the unstable / liquid-loading limb.

    Reports the turning-point rate so the user can decide whether to trim.
    """
    flo = table.axes['FLO'].values
    if flo.size < 3:
        return []
    findings: list[Finding] = []
    for ti in range(table.axes['THP'].values.size):
        for w in range(table.axes['WFR'].values.size):
            for g in range(table.axes['GFR'].values.size):
                for a in range(table.axes['ALQ'].values.size):
                    curve = table.data[ti, w, g, a, :]
                    d = np.sign(np.diff(curve))
                    turns = np.where(np.diff(d) != 0)[0]
                    if turns.size:
                        k = int(turns[0]) + 1
                        if len(findings) < 20:
                            findings.append(_f(table, 'WARNING', 'UNSTABLE_BRANCH', f"non-monotonic BHP vs {table.axes['FLO'].kind}: turning point at {flo[k]:g} ({table.axes['FLO'].unit}) on THP={ti + 1}, WFR/GFR/ALQ idx {w},{g},{a} — unstable limb", locus={'thp': ti, 'wfr': w, 'gfr': g, 'alq': a, 'flo': int(k)}, plot_hint={'kind': 'turning_point', 'flo': float(flo[k]), 'value': float(curve[k])}))
    return findings

def check_absurd_bhp(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Negative, zero, or absurdly large tabulated values."""
    data = table.data
    hi = ctx.thresholds.get('absurd_bhp_bar', 10000.0)
    if table.unit_system == 'FIELD':
        hi *= 14.5038
    bad = ~np.isfinite(data) | (data <= 0.0) | (data > hi)
    if not bad.any():
        return []
    n = int(bad.sum())
    first = np.argwhere(bad)[0]
    return [_f(table, 'ERROR', 'ABSURD_BHP', f'{n} tabulated value(s) <= 0 or > {hi:g}: first at {dict(zip(AXIS_ORDER, first.tolist(), strict=True))}', locus={'indices': first.tolist()})]

def check_bhp_below_thp(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """BHP below THP: negative total pressure drop over the tubing/branch."""
    if table.tabulated != 'BHP':
        return []
    thp = table.axes['THP'].values
    data = table.data
    diff = data - thp.reshape(-1, 1, 1, 1, 1)
    bad = diff < -1e-09
    if not bad.any():
        return []
    n = int(bad.sum())
    first = np.argwhere(bad)[0]
    return [_f(table, 'WARNING', 'BHP_LT_THP', f'{n} point(s) with BHP below THP (negative total pressure drop), first at {dict(zip(AXIS_ORDER, first.tolist(), strict=True))}', locus={'indices': first.tolist()})]

def check_flat_gradient(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Zero pressure gradient over a whole FLO range (generation artifact)."""
    flo = table.axes['FLO'].values
    if flo.size < 2:
        return []
    scale = max(1.0, float(np.max(np.abs(table.data))))
    tol = 1e-06 * scale
    flat = 0
    first_locus = None
    for ti in range(table.axes['THP'].values.size):
        for w in range(table.axes['WFR'].values.size):
            for g in range(table.axes['GFR'].values.size):
                for a in range(table.axes['ALQ'].values.size):
                    curve = table.data[ti, w, g, a, :]
                    if np.all(np.abs(np.diff(curve)) <= tol):
                        flat += 1
                        first_locus = first_locus or (ti, w, g, a)
    if not flat:
        return []
    return [_f(table, 'WARNING', 'FLAT_GRADIENT', f"{flat} curve(s) with zero pressure gradient over the whole {table.axes['FLO'].kind} range (flat region)", locus={'slice': list(first_locus)})]

_GAS_PHASES = {'GAS', 'GASWAT', 'GASOIL', 'GASWATOIL', 'OILGAS', 'WATGAS'}

_LIQUID_ONLY = {'OIL', 'WAT', 'OILWAT', 'WATOIL'}

_GAS_ONLY = {'GAS'}

def check_flo_phase_mismatch(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """FLO type vs consuming-well phase (context from WELSPECS)."""
    deck = ctx.deck
    if deck is None or not table.consumers.wells:
        return []
    flo = table.axes['FLO'].kind
    findings: list[Finding] = []
    for well in table.consumers.wells:
        phase = getattr(deck, 'well_phases', {}).get(well)
        if not phase:
            continue
        if flo == 'GAS' and phase in _LIQUID_ONLY:
            findings.append(_f(table, 'WARNING', 'FLO_PHASE', f'FLO type GAS on table consumed by {well} (phase {phase}) — gas-rate table on a liquid well', locus={'well': well, 'phase': phase}))
        elif flo in ('OIL', 'LIQ', 'WG', 'TM') and phase in _GAS_ONLY:
            findings.append(_f(table, 'WARNING', 'FLO_PHASE', f'FLO type {flo} on table consumed by {well} (phase {phase}) — liquid-rate table on a gas well', locus={'well': well, 'phase': phase}))
    return findings

def check_alq_length1_with_gaslift(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """ALQ axis of length 1 on a well with gas lift declared (WLIFTOPT)."""
    deck = ctx.deck
    if deck is None or table.axis_lengths['ALQ'] != 1 or (not table.consumers.wells):
        return []
    gl = getattr(deck, 'gaslift_wells', set())
    offenders = [w for w in table.consumers.wells if w in gl]
    if not offenders:
        return []
    return [_f(table, 'WARNING', 'ALQ_GASLIFT', f"ALQ axis has a single value but well(s) {', '.join(offenders)} declare gas lift (WLIFTOPT) — lift response cannot be tabulated", locus={'wells': offenders})]

def check_alq_axis_blank_type(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """ALQ axis present (length > 1) but ALQ type blank."""
    if table.axis_lengths['ALQ'] > 1 and table.axes['ALQ'].kind == '':
        return [_f(table, 'WARNING', 'ALQ_BLANK', 'ALQ axis has multiple values but ALQ type is blank', locus={'alq_length': table.axis_lengths['ALQ']})]
    return []

def check_units_mismatch(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Table unit system differs from the deck's."""
    deck = ctx.deck
    if deck is None:
        return []
    t_units, d_units = (table.unit_system, getattr(deck, 'unit_system', 'DEFAULT'))
    if t_units != 'DEFAULT' and d_units != 'DEFAULT' and (t_units != d_units):
        return [_f(table, 'ERROR', 'UNITS_MISMATCH', f'table declared in {t_units} but deck is {d_units}')]
    return []

def check_datum_mismatch(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Datum depth far from consuming wells' WELSPECS datum."""
    deck = ctx.deck
    if deck is None or not table.consumers.wells:
        return []
    tol = ctx.thresholds.get('datum_tolerance_m', 100.0)
    findings: list[Finding] = []
    for well in table.consumers.wells:
        wd = getattr(deck, 'wells_datum', {}).get(well)
        if wd is None:
            continue
        if abs(table.datum_depth - wd) > tol:
            findings.append(_f(table, 'WARNING', 'DATUM_MISMATCH', f'table datum {table.datum_depth:g} m far from well {well} datum {wd:g} m (tolerance {tol:g} m)', locus={'well': well, 'table_datum': table.datum_depth, 'well_datum': wd}))
    return findings

def check_dead_table(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Table not referenced by any well or branch."""
    if not table.consumers:
        return [_f(table, 'INFO', 'DEAD_TABLE', 'table not referenced by any well or branch')]
    return []

def check_role_conflict(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Table referenced by both a well and a branch (almost always a mistake)."""
    if table.consumers.wells and table.consumers.branches:
        return [_f(table, 'WARNING', 'ROLE_CONFLICT', f"table referenced by well(s) {', '.join(table.consumers.wells)} and branch(es) {', '.join((f'{a}->{b}' for a, b in table.consumers.branches))}", locus={'wells': list(table.consumers.wells), 'branches': [list(b) for b in table.consumers.branches]}, plot_hint={'kind': 'role_conflict'})]
    return []

def check_clamp_fraction(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Fraction of simulated timesteps clamped on each axis, per well."""
    if not ctx.coverage:
        return []
    thr = ctx.thresholds.get('clamp_warning_frac', 0.05)
    findings: list[Finding] = []
    for report in ctx.coverage.get(table.number, []):
        for a in report.axes_with_data:
            frac = report.fraction_clamped(a)
            if frac >= thr:
                findings.append(_f(table, 'WARNING', 'CLAMP_FRACTION', f'well {report.well}: {frac * 100:.1f}% of {report.n_timesteps} timesteps clamped on {a} (low {report.fraction_clamped_low(a) * 100:.1f}%, high {report.fraction_clamped_high(a) * 100:.1f}%)', locus={'well': report.well, 'axis': a, 'fraction': frac}, plot_hint={'kind': 'clamped', 'well': report.well, 'axis': a}))
    return findings

def check_run_max_exceeds(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Run maximum on an axis exceeding the table maximum (report the ratio)."""
    if not ctx.coverage:
        return []
    findings: list[Finding] = []
    for report in ctx.coverage.get(table.number, []):
        for a in report.axes_with_data:
            ratio = report.run_max_over_table(a, table)
            if ratio is not None and ratio > 1.0 + 1e-09:
                findings.append(_f(table, 'WARNING', 'RUN_MAX_EXCEEDS', f'well {report.well}: run maximum on {a} is {ratio:.2f}x the table maximum — table is being extrapolated (clamped)', locus={'well': report.well, 'axis': a, 'ratio': ratio}))
    return findings

def check_persistent_unstable(table: VfpTable, ctx: QCContext) -> list[Finding]:
    """Well operating persistently on the unstable branch (check 10 finding)."""
    if not ctx.coverage or table.axes['FLO'].values.size < 3:
        return []
    flo = table.axes['FLO'].values
    turning_rates: list[float] = []
    for ti in range(table.axes['THP'].values.size):
        for w in range(table.axes['WFR'].values.size):
            for g in range(table.axes['GFR'].values.size):
                for a in range(table.axes['ALQ'].values.size):
                    curve = table.data[ti, w, g, a, :]
                    d = np.sign(np.diff(curve))
                    turns = np.where(np.diff(d) != 0)[0]
                    if turns.size:
                        turning_rates.append(float(flo[int(turns[0]) + 1]))
    if not turning_rates:
        return []
    min_turn = min(turning_rates)
    thr = ctx.thresholds.get('unstable_frac', 0.2)
    findings: list[Finding] = []
    for report in ctx.coverage.get(table.number, []):
        flo_run = report.axis_values.get('FLO')
        if flo_run is None or flo_run.size == 0:
            continue
        frac = float(np.mean(flo_run >= min_turn))
        if frac >= thr:
            findings.append(_f(table, 'WARNING', 'PERSISTENT_UNSTABLE', f"well {report.well} operates {frac * 100:.1f}% of timesteps at or above the unstable-branch turning point ({table.axes['FLO'].kind}={min_turn:g} {table.axes['FLO'].unit})", locus={'well': report.well, 'turning_rate': min_turn, 'fraction': frac}))
    return findings

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/core/qc/engine.py
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckDef:
    check_id: str
    name: str
    severity: str
    description: str
    fn: object

CHECK_REGISTRY: tuple[CheckDef, ...] = (CheckDef('RECORD_COUNT', 'record count', 'ERROR', 'data record count != NTHP x NWFR x NGFR x NALQ (parse-enforced)', check_record_count), CheckDef('RECORD_WIDTH', 'record width', 'ERROR', 'data record with != NFLO values (parse-enforced)', check_record_width), CheckDef('AXIS_MONOTONIC', 'axis monotonic', 'ERROR', 'axis values not strictly increasing (parse-enforced)', check_axes_monotonic), CheckDef('NON_FINITE', 'non-finite values', 'ERROR', 'NaN / inf entries in the hypercube', check_non_finite), CheckDef('INDEX_RANGE', 'index range', 'ERROR', 'data-record index out of range (parse-enforced)', check_indices_in_range), CheckDef('HEADER_ENUM', 'header enums', 'ERROR', 'header type strings outside valid enumerations', check_header_enums), CheckDef('DUP_TABLE', 'duplicate tables', 'ERROR', 'duplicate table numbers in the deck', check_duplicate_tables), CheckDef('THP_MONOTONIC', 'BHP vs THP monotonic', 'ERROR', 'BHP must increase with THP at fixed other axes', check_thp_monotonic), CheckDef('CROSSING', 'crossing THP curves', 'ERROR', 'THP curves cross at fixed other axes', check_crossing_thp_curves), CheckDef('UNSTABLE_BRANCH', 'unstable branch', 'WARNING', 'non-monotonic BHP vs FLO (liquid-loading limb)', check_unstable_branch), CheckDef('ABSURD_BHP', 'absurd BHP', 'ERROR', 'negative, zero or absurdly large tabulated values', check_absurd_bhp), CheckDef('BHP_LT_THP', 'BHP below THP', 'WARNING', 'negative total pressure drop over tubing/branch', check_bhp_below_thp), CheckDef('FLAT_GRADIENT', 'flat gradient', 'WARNING', 'zero pressure gradient over a whole FLO range', check_flat_gradient), CheckDef('FLO_PHASE', 'FLO vs fluid system', 'WARNING', 'FLO type contradicts consuming well phase', check_flo_phase_mismatch), CheckDef('ALQ_GASLIFT', 'ALQ vs gas lift', 'WARNING', 'ALQ axis length 1 on a well with WLIFTOPT', check_alq_length1_with_gaslift), CheckDef('ALQ_BLANK', 'blank ALQ type', 'WARNING', 'ALQ axis present but ALQ type blank', check_alq_axis_blank_type), CheckDef('UNITS_MISMATCH', 'unit system', 'ERROR', "table unit system differs from the deck's", check_units_mismatch), CheckDef('DATUM_MISMATCH', 'datum depth', 'WARNING', "datum far from consuming well's WELSPECS datum", check_datum_mismatch), CheckDef('DEAD_TABLE', 'dead table', 'INFO', 'table not referenced by any well or branch', check_dead_table), CheckDef('ROLE_CONFLICT', 'well/branch conflict', 'WARNING', 'table referenced by both a well and a branch', check_role_conflict), CheckDef('CLAMP_FRACTION', 'clamped timesteps', 'WARNING', 'fraction of timesteps clamped on an axis above threshold', check_clamp_fraction), CheckDef('RUN_MAX_EXCEEDS', 'run max vs table max', 'WARNING', 'run maximum exceeds the table maximum (extrapolation)', check_run_max_exceeds), CheckDef('PERSISTENT_UNSTABLE', 'persistent unstable branch', 'WARNING', 'well operating persistently on the unstable branch', check_persistent_unstable))

_CHECK_BY_ID = {c.check_id: c for c in CHECK_REGISTRY}

def check_summary(findings: list[Finding]) -> tuple[int, int, int]:
    """(n_error, n_warning, n_info) counts."""
    n_err = sum(1 for f in findings if f.severity == 'ERROR')
    n_warn = sum(1 for f in findings if f.severity == 'WARNING')
    n_info = sum(1 for f in findings if f.severity == 'INFO')
    return (n_err, n_warn, n_info)

def run_qc(deck, context: QCContext | None=None) -> list[Finding]:
    """All findings for a deck: parse/ref findings + every registry check.

    Deduped by (check_id, table_number, message); deterministic order.
    """
    ctx = context or QCContext(deck=deck)
    out: list[Finding] = list(deck.findings)
    seen = {(f.check_id, f.table_number, f.message) for f in out}
    for no in deck.table_order:
        table: VfpTable = deck.tables[no]
        for check in CHECK_REGISTRY:
            for f in check.fn(table, ctx):
                key = (f.check_id, f.table_number, f.message)
                if key not in seen:
                    seen.add(key)
                    out.append(f)
    out.sort(key=lambda f: (f.table_number, f.check_id))
    return out

def worst_severity(findings: list[Finding]) -> str | None:
    order = {'INFO': 0, 'WARNING': 1, 'ERROR': 2}
    if not findings:
        return None
    return max((f.severity for f in findings), key=lambda s: order[s])

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/viz/figures.py
# -----------------------------------------------------------------------------

_MAX_POINTS_PER_TRACE = 400

_COLOR_SEQ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def _display_axes(table: VfpTable) -> list[str]:
    """Axes with more than one value; degenerate (length-1) axes are hidden."""
    return [a for a in AXIS_ORDER if table.axis_lengths[a] > 1]

def _fmt(v: float, unit: str='') -> str:
    s = f'{v:g}'
    return f'{s} {unit}'.strip()

def _slice_and_reorder(table: VfpTable, x_axis: str, family: str, fixed: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a 2-D slab (family, x) with axis value vectors.

    ``fixed`` holds 0-based indices for every other axis (defaults applied by
    the caller for degenerate axes).
    """
    if x_axis == family:
        raise ValueError('x_axis and family must differ')
    slices = []
    for a in AXIS_ORDER:
        if a in (x_axis, family):
            slices.append(slice(None))
        else:
            slices.append(int(fixed.get(a, 0)))
    arr = table.data[tuple(slices)]
    dims = [a for a in AXIS_ORDER if a in (x_axis, family)]
    arr = arr.transpose([dims.index(family), dims.index(x_axis)])
    return (arr, table.axes[x_axis].values, table.axes[family].values)

def _add_trace(fig, x, y, name, color, showlegend=True, dash=None):
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name, showlegend=showlegend, line=dict(color=color, dash=dash) if dash else dict(color=color), marker=dict(size=5), hovertemplate=f'<b>{name}</b><br>%{{x:.6g}}<br>%{{y:.6g}}<extra></extra>'))

def _axis_label(table: VfpTable, axis: str) -> str:
    ax = table.axes[axis]
    unit = ax.unit or ''
    kind = ax.kind or axis
    if axis == 'THP' and table.role == 'BRANCH':
        base = 'outlet pressure'
    elif axis == 'FLO' and table.role == 'BRANCH':
        base = 'throughput'
    elif axis == 'THP':
        base = 'wellhead pressure'
    elif axis == 'FLO':
        base = 'rate'
    else:
        base = axis
    return f'{base} [{kind}, {unit}]' if unit else f'{base} [{kind}]'

def _y_label(table: VfpTable) -> str:
    unit = resolve_axis_unit(table.unit_system, 'THP', 'THP')
    qty = 'inlet pressure' if table.role == 'BRANCH' else table.tabulated
    return f'{qty} [{unit}]' if unit else qty

def lift_curve_figure(table: VfpTable, x_axis: str='FLO', family: str='THP', fixed: dict[str, int] | None=None, downsample: int=_MAX_POINTS_PER_TRACE, shade_coverage: bool=True) -> go.Figure:
    """One curve per ``family`` value; remaining axes fixed at given indices.

    x_axis and family can be any of the table's non-degenerate axes
    (generalised axis pivot, M3). Hover shows the exact tabulated values.
    """
    axes = _display_axes(table)
    if x_axis not in axes or family not in axes:
        raise ValueError(f'x_axis={x_axis}, family={family} must be non-degenerate axes of the table (available: {axes})')
    fixed = dict(fixed or {})
    for a in axes:
        fixed.setdefault(a, 0)
    arr, xvals, famvals = _slice_and_reorder(table, x_axis, family, fixed)
    nfam, nx = arr.shape
    fig = go.Figure()
    fig.update_layout(title=f'{table.label} — {family} curves @ ' + ', '.join(f'{a}={table.axes[a].values[fixed[a]]:g}' for a in axes if a not in (x_axis, family)), xaxis_title=_axis_label(table, x_axis), yaxis_title=_y_label(table), hovermode='closest', template='plotly_white', legend_title=family)
    unit = table.axes[family].unit
    stride = max(1, int(np.ceil(nx / downsample))) if downsample else 1
    xs = xvals[::stride]
    for fi in range(nfam):
        ys = arr[fi][::stride]
        _add_trace(fig, xs, ys, f'{family}={_fmt(famvals[fi], unit)}', _COLOR_SEQ[fi % len(_COLOR_SEQ)])
    if shade_coverage and table.axes[x_axis].values.size > 1:
        lo, hi = (float(xvals[0]), float(xvals[-1]))
        fig.add_vrect(x0=lo - (hi - lo), x1=lo, fillcolor='red', opacity=0.04, line_width=0)
        fig.add_vrect(x0=hi, x1=hi + (hi - lo), fillcolor='red', opacity=0.04, line_width=0)
    return fig

def branch_figure(table: VfpTable, x_axis: str='FLO', family: str='THP', fixed: dict[str, int] | None=None, downsample: int=_MAX_POINTS_PER_TRACE) -> go.Figure:
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
    if family == 'THP':
        outlet = np.broadcast_to(famvals[:, None], arr.shape)
    elif x_axis == 'THP':
        outlet = np.broadcast_to(xvals[None, :], arr.shape)
    else:
        outlet = np.full(arr.shape, float(table.axes['THP'].values[fixed.get('THP', 0)]))
    dp = arr - outlet
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=('inlet pressure vs throughput', 'delta-P vs throughput'))
    unit = table.axes[family].unit
    stride = max(1, int(np.ceil(nx / downsample))) if downsample else 1
    xs = xvals[::stride]
    colors = [_COLOR_SEQ[fi % len(_COLOR_SEQ)] for fi in range(nfam)]
    names = [f'{family}={_fmt(famvals[fi], unit)}' for fi in range(nfam)]
    for fi in range(nfam):
        fig.add_trace(go.Scatter(x=xs, y=arr[fi][::stride], mode='lines+markers', name=names[fi], line=dict(color=colors[fi]), marker=dict(size=5), hovertemplate=f'<b>{names[fi]}</b><br>rate: %{{x:.6g}}<br>inlet: %{{y:.6g}}<extra></extra>'), row=1, col=1)
    for fi in range(nfam):
        fig.add_trace(go.Scatter(x=xs, y=dp[fi][::stride], mode='lines+markers', name=names[fi], line=dict(color=colors[fi], dash='dot'), marker=dict(size=5), showlegend=False, hovertemplate=f'<b>{names[fi]}</b><br>rate: %{{x:.6g}}<br>dP: %{{y:.6g}}<extra></extra>'), row=2, col=1)
    unit_y = resolve_axis_unit(table.unit_system, 'THP', 'THP')
    fig.update_xaxes(title_text=_axis_label(table, x_axis), row=2, col=1)
    fig.update_xaxes(title_text='', row=1, col=1)
    fig.update_yaxes(title_text=f'inlet pressure [{unit_y}]' if unit_y else 'inlet pressure', row=1, col=1)
    fig.update_yaxes(title_text=f'delta-P [{unit_y}]' if unit_y else 'delta-P', row=2, col=1)
    fig.update_layout(title=table.label, template='plotly_white', hovermode='closest', legend_title=family)
    return fig

def heatmap_figure(table: VfpTable, x_axis: str, y_axis: str, fixed: dict[str, int] | None=None, as_contour: bool=True) -> go.Figure:
    """Contour (default) or 3-D surface of the tabulated value over two axes."""
    axes = _display_axes(table)
    if x_axis not in axes or y_axis not in axes or x_axis == y_axis:
        raise ValueError(f'x_axis/y_axis must be distinct non-degenerate axes (available: {axes})')
    fixed = dict(fixed or {})
    for a in axes:
        fixed.setdefault(a, 0)
    slices = []
    for a in AXIS_ORDER:
        if a in (x_axis, y_axis):
            slices.append(slice(None))
        else:
            slices.append(int(fixed.get(a, 0)))
    z = table.data[tuple(slices)]
    dims = [a for a in AXIS_ORDER if a in (x_axis, y_axis)]
    z = z.transpose([dims.index(y_axis), dims.index(x_axis)])
    xvals = table.axes[x_axis].values
    yvals = table.axes[y_axis].values
    if as_contour:
        fig = go.Figure(go.Contour(z=z, x=xvals, y=yvals, colorscale='Viridis', colorbar=dict(title=_y_label(table)), hovertemplate=f'{x_axis}: %{{x:.6g}}<br>{y_axis}: %{{y:.6g}}<br>value: %{{z:.6g}}<extra></extra>'))
    else:
        fig = go.Figure(go.Surface(z=z, x=xvals, y=yvals, colorscale='Viridis', colorbar=dict(title=_y_label(table))))
    fixed_desc = ', '.join(f'{a}={table.axes[a].values[fixed[a]]:g}' for a in axes if a not in (x_axis, y_axis))
    fig.update_layout(title=f'{table.label} — {_y_label(table)} over {y_axis} vs {x_axis}' + (f' @ {fixed_desc}' if fixed_desc else ''), xaxis_title=_axis_label(table, x_axis), yaxis_title=_axis_label(table, y_axis), template='plotly_white')
    return fig

def apply_plot_hint(fig: go.Figure, hint: dict | None) -> None:
    """Highlight the problem locus described by a Finding.plot_hint."""
    if not hint:
        return
    kind = hint.get('kind')
    if kind == 'turning_point' and 'flo' in hint and ('value' in hint):
        fig.add_trace(go.Scatter(x=[hint['flo']], y=[hint['value']], mode='markers', marker=dict(size=14, color='red', symbol='x'), name='turning point', hovertemplate='turning point: %{x:.6g}<extra></extra>'))
    elif kind == 'crossing' and 'x' in hint and ('y' in hint):
        fig.add_trace(go.Scatter(x=hint['x'], y=hint['y'], mode='markers', marker=dict(size=10, color='red', symbol='circle-open'), name='crossing', hovertemplate='curve crossing<extra></extra>'))
    elif kind == 'clamped' and 'x' in hint and ('y' in hint):
        fig.add_trace(go.Scatter(x=hint['x'], y=hint['y'], mode='markers', marker=dict(size=6, color='red', opacity=0.6), name='clamped timesteps', hovertemplate='clamped: %{x:.6g}, %{y:.6g}<extra></extra>'))

def coverage_overlay_figure(table, report) -> go.Figure:
    """Operating envelope vs table: clamped scatter + per-axis coverage bars.

    Top: run points (FLO vs WBHP, or FLO vs THP when WBHP is missing),
    coloured red when any axis was clamped; table's first THP curves drawn as
    reference. Bottom: per-axis stacked bars (in-range / clamped low / high).
    """
    flo = report.axis_values.get('FLO')
    if flo is None:
        raise ValueError('coverage report has no FLO data')
    bhp = report.bhp if report.bhp is not None else report.axis_values.get('THP')
    any_clamped = np.zeros(flo.size, dtype=bool)
    for a in report.axes_with_data:
        any_clamped |= report.clamped_low[a] | report.clamped_high[a]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=('run envelope vs table (clamped points in red)', 'per-axis coverage (fraction of timesteps)'))
    unit = table.axes['FLO'].unit
    fig.add_trace(go.Scatter(x=flo, y=bhp, mode='markers', marker=dict(size=5, color=['#c0392b' if c else '#1f77b4' for c in any_clamped], opacity=0.7), name='run (clamped=red)', hovertemplate='FLO: %{x:.6g}<br>BHP: %{y:.6g}<extra></extra>'), row=1, col=1)
    try:
        ref = lift_curve_figure(table, x_axis='FLO', family='THP', fixed={'WFR': 0, 'GFR': 0, 'ALQ': 0})
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
    fig.add_trace(go.Bar(x=axes, y=ok, name='in range', marker_color='#1e8449'), row=2, col=1)
    fig.add_trace(go.Bar(x=axes, y=low, name='clamped low', marker_color='#b9770e'), row=2, col=1)
    fig.add_trace(go.Bar(x=axes, y=high, name='clamped high', marker_color='#c0392b'), row=2, col=1)
    fig.update_layout(barmode='stack', title=f'{table.label} — well {report.well} coverage ({report.n_timesteps} timesteps)', template='plotly_white', xaxis2_title=f'rate [{unit}]' if unit else 'rate', yaxis2_title='fraction', hovermode='closest')
    return fig

def compare_figure(tables: list, x_axis: str='FLO', family: str='THP', fixed: dict[str, int] | None=None) -> go.Figure:
    """Overlay two+ tables on the same axes; unit-system guard per spec 3.5.

    Refuses (ValueError) when tables disagree on unit system or on the
    FLO/WFR/GFR/ALQ axis *types* — no silent conversion, no apples/oranges.
    """
    if len(tables) < 2:
        raise ValueError('compare needs at least two tables')
    t0 = tables[0]
    systems = {t.unit_system for t in tables}
    if len(systems) > 1:
        raise ValueError(f'cannot overlay tables with differing unit systems ({systems}); convert explicitly first')
    for axis in ('FLO', 'WFR', 'GFR', 'ALQ'):
        kinds = {t.axes[axis].kind for t in tables}
        if len(kinds) > 1:
            raise ValueError(f'cannot overlay tables with differing {axis} types {kinds}')
    fig = go.Figure()
    unit = t0.axes[family].unit
    for k, t in enumerate(tables):
        arr, xvals, famvals = _slice_and_reorder(t, x_axis, family, fixed or {})
        label = f'table {t.number}' + (f" ({t.label.split('(')[-1][:-1]})" if False else '')
        for fi in range(arr.shape[0]):
            fig.add_trace(go.Scatter(x=xvals, y=arr[fi], mode='lines+markers', name=f'{label} THP={_fmt(famvals[fi], unit)}', line=dict(color=_COLOR_SEQ[(k * 7 + fi) % len(_COLOR_SEQ)]), marker=dict(size=4), hovertemplate=f'{label}: %{{x:.6g}}, %{{y:.6g}}<extra></extra>'))
    fig.update_layout(title=' vs '.join(f'table {t.number}' for t in tables), xaxis_title=_axis_label(t0, x_axis), yaxis_title=_y_label(t0), template='plotly_white', hovermode='closest')
    return fig

def compare_difference_figure(table_a, table_b, x_axis: str='FLO') -> go.Figure:
    """Difference panel: table_A minus table_B interpolated onto A's grid."""
    a = table_a.data
    b = table_b.data
    if a.shape != b.shape:
        raise ValueError(f'difference panel requires equal axis grids; shapes {a.shape} vs {b.shape}')
    fig = go.Figure()
    for ti in range(min(a.shape[0], 8)):
        fig.add_trace(go.Scatter(x=table_a.axes[x_axis].values, y=a[ti, 0, 0, 0, :] - b[ti, 0, 0, 0, :], mode='lines+markers', name=f"THP={table_a.axes['THP'].values[ti]:g}", hovertemplate='%{x:.6g}, d(BHP)=%{y:.6g}<extra></extra>'))
    fig.update_layout(title=f'table {table_a.number} − table {table_b.number} (BHP difference)', xaxis_title=_axis_label(table_a, x_axis), yaxis_title='difference', template='plotly_white', hovermode='closest')
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
        fig.add_trace(go.Scatter(x=[x0, x1], y=[0, 0], mode='lines+markers', line=dict(color='#7f7f7f', width=2), marker=dict(size=1), text=[f'{dt}->{ut} (VFP {vfp})', ''], hoverinfo='text', name=f'{dt}->{ut}'))
    fig.add_trace(go.Scatter(x=[pos[n] for n in nodes], y=[0] * len(nodes), mode='markers+text', marker=dict(size=[26 if n in fixed else 16 for n in nodes], color=['#c0392b' if n in fixed else '#1f77b4' for n in nodes]), text=nodes, textposition='bottom center', hovertemplate='%{text}<extra></extra>'))
    fig.update_layout(title='network topology (red = fixed pressure)', template='plotly_white', showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), height=320)
    return fig

def build_report_html(deck, out) -> None:
    """Self-contained HTML: every table's figure + QC verdicts."""
    import html as _html
    sections: list[str] = []
    all_findings = run_qc(deck)
    first = True
    for no in deck.table_order:
        t = deck.tables[no]
        fig = branch_figure(t) if t.role == 'BRANCH' else lift_curve_figure(t)
        findings = [f for f in all_findings if f.table_number == no]
        worst = max((SEVERITY_ORDER[f.severity] for f in findings), default=0)
        badge = {2: 'ERROR', 1: 'WARNING', 0: 'OK'}[worst]
        color = {'ERROR': '#c0392b', 'WARNING': '#b9770e', 'OK': '#1e8449'}[badge]
        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn' if first else False, config={'displayModeBar': True})
        first = False
        meta = f'<p>kind: {t.kind} · role: {t.role} · datum: {t.datum_depth:g} · units: {t.unit_system} · axes: ' + ', '.join(f'{a}={t.axes[a].kind}({t.axis_lengths[a]})' for a in AXIS_ORDER) + '</p>'
        consumers = []
        if t.consumers.wells:
            consumers.append('wells: ' + ', '.join(t.consumers.wells))
        if t.consumers.branches:
            consumers.append('branches: ' + ', '.join((f'{a}->{b}' for a, b in t.consumers.branches)))
        if consumers:
            meta += '<p>' + ' · '.join(consumers) + '</p>'
        fl = ''.join(f"<li style='color:{ {'ERROR': '#c0392b', 'WARNING': '#b9770e', 'INFO': '#7f8c8d'}[f.severity]}'><b>{f.check_id}</b> ({f.severity}): {_html.escape(f.message)}</li>" for f in findings)
        findings_html = f'<ul>{fl}</ul>' if fl else '<p>no findings</p>'
        sections.append(f"<section><h2 style='color:{color}'>[{badge}] {_html.escape(t.label)}</h2>{meta}{fig_html}<h3>QC findings</h3>{findings_html}</section>")
    doc = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>VFPScope report</title><style>body{{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}}h2{{border-bottom:2px solid #ddd;padding-bottom:.2em}}section{{margin-bottom:3em}}</style></head><body><h1>VFPScope report — {_html.escape(str(deck.path))}</h1>" + ''.join(sections) + '</body></html>'
    out.write_text(doc, encoding='utf-8')

# -----------------------------------------------------------------------------
# Flattened from src/vfpscope/app/main.py
# -----------------------------------------------------------------------------

PRESETS = {'BHP vs FLO @ THP (well)': ('FLO', 'THP'), 'BHP vs WFR @ FLO': ('WFR', 'FLO'), 'THP vs FLO @ WFR (network)': ('FLO', 'WFR'), 'BHP vs ALQ @ FLO (gas lift)': ('ALQ', 'FLO')}

@st.cache_resource(show_spinner='Parsing deck...')
def parse_deck(path: str):
    return load_deck(path)

def _deck_path_from_args() -> str | None:
    env = os.environ.get('VFPSCOPE_DECK')
    if env:
        return env
    args = sys.argv[1:]
    if '--deck' in args:
        i = args.index('--deck')
        if i + 1 < len(args):
            return args[i + 1]
    return None

def _table_options(deck) -> list[str]:
    out = []
    for no in deck.table_order:
        t = deck.tables[no]
        n_err = sum(1 for f in deck.findings if f.table_number == no and f.severity == 'ERROR')
        n_warn = sum(1 for f in deck.findings if f.table_number == no and f.severity == 'WARNING')
        badge = ''
        if n_err:
            badge = ' [red]✗[/]'
        elif n_warn:
            badge = ' [yellow]![/]'
        consumers = []
        if t.consumers.wells:
            consumers.append(f'{len(t.consumers.wells)} wells')
        if t.consumers.branches:
            consumers.append(f'{len(t.consumers.branches)} branches')
        cons = f" ({', '.join(consumers)})" if consumers else ''
        out.append(f'#{no} {t.kind} {t.role}{badge}{cons}')
    return out

def main() -> None:
    st.set_page_config(page_title='VFPScope', layout='wide', page_icon='📈')
    st.title('VFPScope — VFP table visualizer')
    arg_deck = _deck_path_from_args()
    deck_path = st.sidebar.text_input('Deck / include path', value=arg_deck or '')
    if not deck_path:
        st.info('Enter a deck path (or run `vfpscope serve <deck>`).')
        return
    if not os.path.exists(deck_path):
        st.error(f'file not found: {deck_path}')
        return
    try:
        deck = parse_deck(str(Path(deck_path).resolve()))
    except VfpParseError as e:
        st.error(str(e))
        return
    st.sidebar.caption(f"{len(deck.tables)} table(s), {sum(1 for f in deck.findings if f.severity == 'ERROR')} error(s), {sum(1 for f in deck.findings if f.severity == 'WARNING')} warning(s)")
    options = _table_options(deck)
    sel = st.sidebar.selectbox('Table', options, index=0)
    table_no = int(sel.split(' ')[0].lstrip('#'))
    table = deck.tables[table_no]
    tab_curves, tab_heatmap, tab_qc, tab_compare, tab_coverage, tab_network = st.tabs(['Curves', 'Heatmap', 'QC', 'Compare', 'Coverage', 'Network'])
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
    with tab_network:
        _view_network(deck, table)

def _fixed_sliders(table, exclude: set[str], key_prefix: str='s'):
    """Sliders for every non-degenerate axis not used as x/family (M3)."""
    fixed: dict[str, int] = {}
    for a in ('THP', 'WFR', 'GFR', 'ALQ', 'FLO'):
        vals = table.axes[a].values
        if vals.size <= 1 or a in exclude:
            continue
        i = st.sidebar.slider(f'{a} ({table.axes[a].kind})', min_value=0, max_value=vals.size - 1, value=0, key=f'{key_prefix}_{table.number}_{a}')
        fixed[a] = int(i)
    return fixed

def _view_curves(deck, table) -> None:
    axes = [a for a in ('FLO', 'THP', 'WFR', 'GFR', 'ALQ') if table.axis_lengths[a] > 1]
    preset = st.sidebar.selectbox('Preset', list(PRESETS), index=0)
    px, pfamily = PRESETS[preset]
    x_axis = st.sidebar.selectbox('x axis', axes, index=axes.index(px) if px in axes else 0)
    family = st.sidebar.selectbox('curve family', [a for a in axes if a != x_axis], index=0)
    fixed = _fixed_sliders(table, exclude={x_axis, family}, key_prefix='cv')
    show_hints = st.sidebar.checkbox('Show QC plot hints', value=False)
    if table.role == 'BRANCH':
        fig = branch_figure(table, x_axis=x_axis, family=family, fixed=fixed)
    else:
        fig = lift_curve_figure(table, x_axis=x_axis, family=family, fixed=fixed)
    if show_hints:
        for f in deck.findings:
            if f.table_number == table.number:
                apply_plot_hint(fig, f.plot_hint)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander('Table info'):
        st.json({'number': table.number, 'kind': table.kind, 'role': table.role, 'datum_depth': table.datum_depth, 'unit_system': table.unit_system, 'tabulated': table.tabulated, 'axis_lengths': table.axis_lengths, 'axis_kinds': {a: table.axes[a].kind for a in table.axes}, 'consumers': {'wells': list(table.consumers.wells), 'branches': [list(b) for b in table.consumers.branches]}, 'source': str(table.source)})

def _view_heatmap(deck, table) -> None:
    axes = [a for a in ('FLO', 'THP', 'WFR', 'GFR', 'ALQ') if table.axis_lengths[a] > 1]
    if len(axes) < 2:
        st.info('Need at least two non-degenerate axes for a heatmap.')
        return
    x_axis = st.sidebar.selectbox('heatmap x', axes, index=0)
    y_axis = st.sidebar.selectbox('heatmap y', [a for a in axes if a != x_axis], index=0)
    fixed = _fixed_sliders(table, exclude={x_axis, y_axis}, key_prefix='hm')
    as_contour = st.sidebar.checkbox('Contour (vs 3-D surface)', value=True)
    st.plotly_chart(heatmap_figure(table, x_axis=x_axis, y_axis=y_axis, fixed=fixed, as_contour=as_contour), use_container_width=True)

def _view_qc(deck, table) -> None:
    findings = [f for f in deck.findings if f.table_number == table.number]
    all_findings = [f for f in deck.findings if f.table_number == 0]
    if not findings and (not all_findings):
        st.success('No QC findings for this table.')
        return
    color = {'ERROR': '🔴', 'WARNING': '🟡', 'INFO': '⚪'}
    for f in sorted(findings + all_findings, key=lambda x: -SEVERITY_ORDER[x.severity]):
        locus = f'  \n`{f.locus}`' if f.locus else ''
        st.markdown(f'**{color[f.severity]} {f.check_id}** ({f.severity}) — {f.message}{locus}')
    st.caption('Run `vfpscope qc <deck> --fail-on warning` in a terminal for exit-code gating.')

def _view_compare(deck, table) -> None:
    st.markdown('Overlay a second table on the same axes (unit-system guard active).')
    other = st.sidebar.text_input('Second deck / include path (compare)')
    if not other or not os.path.exists(other):
        st.info('Enter a second deck path to compare.')
        return
    try:
        other_deck = parse_deck(str(Path(other).resolve()))
    except VfpParseError as e:
        st.error(str(e))
        return
    opts = [f'#{no} {other_deck.tables[no].kind}' for no in other_deck.table_order]
    other_no = int(st.sidebar.selectbox('Second table', opts, index=0).split(' ')[0].lstrip('#'))
    other_table = other_deck.tables[other_no]
    try:
        fig = compare_figure([table, other_table])
        st.plotly_chart(fig, use_container_width=True)
        try:
            dfig = compare_difference_figure(table, other_table)
            st.plotly_chart(dfig, use_container_width=True)
        except ValueError as e:
            st.warning(f'Difference panel skipped: {e}')
    except ValueError as e:
        st.error(str(e))

def _view_coverage(deck, table) -> None:
    st.markdown('Coverage vs simulation output: paste the path of a `.UNSMRY` file (requires `resdata`).')
    smry = st.sidebar.text_input('UNSMRY path (coverage)')
    if not smry or not os.path.exists(smry):
        st.info('Enter a summary path to compute coverage.')
        return
    try:
        reports_by_table = coverage_for_deck(deck, smry)
    except CoverageUnavailable as e:
        st.error(str(e))
        return
    reports = reports_by_table.get(table.number, [])
    if not reports:
        st.info(f'No coverage vectors found for table {table.number} in that run.')
        return
    for rep in reports:
        st.plotly_chart(coverage_overlay_figure(table, rep), use_container_width=True)
        st.caption('per-axis clamped fractions: ' + ', '.join(f'{a} {rep.fraction_clamped(a) * 100:.1f}%' for a in rep.axes_with_data))

def _view_network(deck, table) -> None:
    if not deck.branches:
        st.info('No BRANPROP branches in this deck.')
        return
    st.plotly_chart(network_graph_figure(deck), use_container_width=True)
    st.markdown('Branches (click-through via the Curves tab):')
    for dt, ut, vfp in deck.branches:
        t = deck.tables.get(vfp)
        label = f'VFP {vfp}' + (f" — {t.axes['FLO'].kind} {t.axes['FLO'].values[-1]:g} {t.axes['FLO'].unit}" if t else '')
        st.markdown(f'- **{dt} → {ut}**: {label}')

if __name__ == "__main__":
    main()
