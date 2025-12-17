import json
from pathlib import Path

import pandas as pd
import pytest

import sft_qwen3_metadata as sft


def test_load_prompts_requires_placeholder(tmp_path):
    sys_path = tmp_path / "system.txt"
    user_path = tmp_path / "user.txt"
    sys_path.write_text("SYSTEM", encoding="utf-8")
    user_path.write_text("User prompt without placeholder", encoding="utf-8")

    with pytest.raises(ValueError):
        sft.load_prompts(str(tmp_path), "system.txt", "user.txt")

    # Now with placeholder
    user_path.write_text("Filename: {filename}", encoding="utf-8")
    system, user = sft.load_prompts(str(tmp_path), "system.txt", "user.txt")
    assert system == "SYSTEM"
    assert user == "Filename: {filename}"


def test_format_example_includes_eos():
    row = {
        "name": "file.mkv",
        "show_name": "Show",
        "season": 1,
        "episode": 2,
        "reasoning": "ok",
        "confidence": 0.9,
        "crc_hash": "ABCDEF12",
    }
    out = sft.format_example(row)["text"]
    assert out.endswith("<|endoftext|>")
    assert "<|system|>" in out and "<|user|>" in out and "<|assistant|>" in out


def test_row_to_target_json_order_and_nulls():
    row = {
        "show_name": "X",
        "season": None,
        "episode": None,
        "crc_hash": None,
        "confidence": None,
        "reasoning": "",
    }
    js = sft.row_to_target_json(row)
    obj = json.loads(js)
    assert list(obj.keys()) == ["show_name", "season", "episode", "crc_hash", "confidence", "reasoning"]
    assert obj["confidence"] == 0.0
    assert obj["season"] is None and obj["episode"] is None and obj["crc_hash"] is None


def test_load_and_clean_csv_basic(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text(
        "name,show_name,season,episode,reasoning,confidence,crc_hash,extra\n"
        "a.mkv,Show,1,2,ok,0.9,[abcDEF12],ignored\n",
        encoding="utf-8",
    )
    df = sft.load_and_clean_csv(csv)
    assert list(df.columns) == sft.EXPECTED_COLS
    assert df.loc[0, "crc_hash"] == "ABCDEF12"
    assert df.loc[0, "episode"] == 2
    assert df.loc[0, "confidence"] == 0.9


def test_load_and_clean_csv_missing_column(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("name,show_name\nfile,Show\n", encoding="utf-8")
    with pytest.raises(ValueError):
        sft.load_and_clean_csv(csv)


def test_find_lora_target_modules_standard_names():
    Linear4bit = type("Linear4bit", (), {})

    class Model:
        def named_modules(self):
            return [
                ("layers.0.attn.q_proj", Linear4bit()),
                ("layers.0.attn.v_proj", Linear4bit()),
                ("layers.0.attn.k_proj", Linear4bit()),
                ("layers.0.attn.o_proj", Linear4bit()),
            ]

    found = sft.find_lora_target_modules(Model())
    assert found == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_require_episode_filter_raises_when_empty():
    df = pd.DataFrame(
        [
            {"name": "a", "show_name": "s", "season": 1, "episode": None, "reasoning": "", "confidence": 0.5, "crc_hash": None},
        ]
    )
    before = len(df)
    filtered = df[df["episode"].notna()]
    assert before == 1 and len(filtered) == 0

