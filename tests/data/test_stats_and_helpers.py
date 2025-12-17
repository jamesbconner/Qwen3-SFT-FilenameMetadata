import pandas as pd
import pytest

import sft_qwen3_metadata as sft


def test_print_descriptive_stats(capsys):
    df = pd.DataFrame(
        [
            {"name": "a", "show_name": "s", "season": 1, "episode": 1, "reasoning": "ok", "confidence": 0.5, "crc_hash": "ABCDEF12"},
            {"name": "b", "show_name": "s2", "season": pd.NA, "episode": pd.NA, "reasoning": "ok2", "confidence": 0.2, "crc_hash": pd.NA},
        ]
    )
    sft.print_descriptive_stats(df)
    out = capsys.readouterr().out
    assert "rows:" in out
    assert "episode stats" in out


def test_extract_first_balanced_json_no_brace():
    obj, err = sft.extract_first_balanced_json("no json here")
    assert obj is None
    assert "no '{'" in err


def test_extract_first_balanced_json_unbalanced():
    obj, err = sft.extract_first_balanced_json("{abc")
    assert obj is None
    assert "unbalanced" in err

