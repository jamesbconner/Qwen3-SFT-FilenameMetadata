import pandas as pd

import sft_qwen3_metadata as sft


def test_detect_data_issues_flags_all():
    long_name = "x" * 201
    df = pd.DataFrame(
        [
            {"name": long_name, "show_name": "s", "season": 1, "episode": 1, "reasoning": "", "confidence": 0.5, "crc_hash": "ZZZZZZZZ"},
            {"name": "dup", "show_name": "s2", "season": 1, "episode": pd.NA, "reasoning": "", "confidence": 0.5, "crc_hash": None},
            {"name": "dup", "show_name": "s3", "season": 1, "episode": 2, "reasoning": "", "confidence": 0.5, "crc_hash": "ABCDEF12"},
        ]
    )
    issues = sft.detect_data_issues(df, max_name_len=200)
    assert issues["long_filenames"] == [0]
    assert set(issues["duplicates"]) == {1, 2}
    assert issues["bad_crc_format"] == [0]
    assert issues["missing_episode"] == [1]

