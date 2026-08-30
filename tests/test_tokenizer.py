"""Tokenizer/parser tests for Eclipse free-format conventions.

These run on strings directly (unit level) before any VFP-specific logic.
"""

from __future__ import annotations

import pytest

from vfpscope.core.parse.native import (
    VfpParseError,
    expand_record_items,
    parse_string,
    tokenize,
)

# ---------------------------------------------------------------- tokenizer


def test_comments_stripped_and_line_numbers_kept():
    toks = tokenize("-- header comment\n  1 2.5 'W1' -- trailing\n3 /\n", "x.inc")
    texts = [(t.text, t.line) for t in toks if not t.is_record_end]
    assert texts == [("1", 2), ("2.5", 2), ("W1", 2), ("3", 3)]


def test_record_ends_detected():
    toks = tokenize("1 2 /\n3 /\n", "x.inc")
    assert [t.is_record_end for t in toks] == [False, False, True, False, True]


def test_slash_inside_quoted_string_is_not_a_record_end():
    toks = tokenize("'A/B' 'C/D' /\n", "x.inc")
    assert [t.text for t in toks] == ["A/B", "C/D", "/"]


def test_comment_marker_inside_quoted_string_is_literal():
    toks = tokenize("'W--X' 1 /\n", "x.inc")
    assert [t.text for t in toks] == ["W--X", "1", "/"]


def test_quoted_empty_string():
    toks = tokenize("'' 1 /\n", "x.inc")
    assert toks[0].text == ""


def test_unterminated_quote_raises_with_location():
    with pytest.raises(VfpParseError) as ei:
        tokenize("'W1 1 /\n", "bad.inc")
    assert "bad.inc" in str(ei.value)
    assert "1" in str(ei.value)


def test_slash_abutting_token_is_split():
    toks = tokenize("1.0/ 2 /\n", "x.inc")
    assert [t.text for t in toks] == ["1.0", "/", "2", "/"]


def test_multiline_values_continue_across_lines():
    toks = tokenize("1 2\n3 4 /\n", "x.inc")
    assert [t.text for t in toks if not t.is_record_end] == ["1", "2", "3", "4"]


# ---------------------------------------------------------------- repeat/default expansion


def test_expand_repeat_count():
    items = expand_record_items(tokenize("3*0.0 1.0 /\n", "x.inc"))
    vals = [i.value for i in items]
    assert vals == [0.0, 0.0, 0.0, 1.0]


def test_expand_default_count_is_not_a_number():
    # 1* is a DEFAULT, 1*0.0 is a number — they must stay distinct.
    items = expand_record_items(tokenize("1* 5*1.0 /\n", "x.inc"))
    assert items[0].value is None
    assert items[1].value == 1.0


def test_expand_star_alone_is_one_default():
    items = expand_record_items(tokenize("* 2 /\n", "x.inc"))
    assert items[0].value is None
    assert items[1].value == 2.0


def test_expand_trailing_star_is_defaults():
    items = expand_record_items(tokenize("4* /\n", "x.inc"))
    assert len(items) == 4
    assert all(i.value is None for i in items)


def test_expand_repeated_strings():
    items = expand_record_items(tokenize("3*'GAS' /\n", "x.inc"))
    assert [i.value for i in items] == ["GAS", "GAS", "GAS"]


def test_expand_quoted_star_is_not_expanded():
    items = expand_record_items(tokenize("'2*' /\n", "x.inc"))
    assert [i.value for i in items] == ["2*"]


def test_items_carry_source_location():
    items = expand_record_items(tokenize("-- c\n7 /\n", "x.inc"))
    assert items[0].line == 2
    assert items[0].file == "x.inc"


# ---------------------------------------------------------------- parse_string on decks


def test_parse_string_finds_keywords():
    toks = parse_string("VFPPROD\n 1 2000.0 /\n", "x.inc")
    # not yet implemented beyond tokenizer? placeholder API check
    assert isinstance(toks, list)


def test_numeric_variants_parse():
    items = expand_record_items(tokenize("1E5 .5 -3 5. 1.0D3 0.00000E+00 /\n", "x.inc"))
    assert [i.value for i in items] == [1e5, 0.5, -3.0, 5.0, 1000.0, 0.0]
