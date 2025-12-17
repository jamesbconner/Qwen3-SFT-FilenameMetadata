import pandas as pd

from sft_qwen3_metadata import _clean_text_series, _to_nullable_int, format_example


def test_to_nullable_int_accepts_intlike_and_rejects_nonint():
    s = pd.Series(["1", "02", "3.0", "4.5", None])
    out = _to_nullable_int(s)
    assert out.tolist() == [1, 2, 3, pd.NA, pd.NA]


def test_clean_text_series_strips_quotes_and_nulls():
    s = pd.Series([' "hello" ', "nan", "NULL", "", None])
    out = _clean_text_series(s)
    assert out.tolist() == ["hello", pd.NA, pd.NA, pd.NA, pd.NA]


def test_format_example_appends_eos():
    row = {
        "name": "file.mkv",
        "show_name": "Show",
        "season": 1,
        "episode": 2,
        "reasoning": "ok",
        "confidence": 0.9,
        "crc_hash": "ABCDEF12",
    }
    formatted = format_example(row)
    assert formatted["text"].endswith("<|endoftext|>")

