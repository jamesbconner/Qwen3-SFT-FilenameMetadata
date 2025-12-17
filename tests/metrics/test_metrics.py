import numpy as np
import pytest
from transformers import EvalPrediction

from sft_qwen3_metadata import (
    compute_metrics,
    extract_first_balanced_json,
)


class DummyTokenizer:
    """Minimal tokenizer stub for metric tests."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        # Accept numpy arrays or lists of strings/ids.
        if isinstance(sequences, np.ndarray):
            sequences = sequences.tolist()
        decoded = []
        for seq in sequences:
            if isinstance(seq, str):
                decoded.append(seq)
            else:
                # Assume seq is list[int]; map to simple chars for testing.
                decoded.append("".join(chr(x) for x in seq if isinstance(x, int)))
        return decoded


def test_extract_first_balanced_json_valid():
    obj, err = extract_first_balanced_json('prefix {"a":1, "b":2} suffix')
    assert err == ""
    assert obj == {"a": 1, "b": 2}


def test_extract_first_balanced_json_unbalanced():
    obj, err = extract_first_balanced_json('prefix {"a":1, "b":2')
    assert obj is None
    assert "unbalanced" in err


def test_compute_metrics_all_pass():
    tok = DummyTokenizer()
    pred = '{"show_name":"A","season":1,"episode":2,"crc_hash":"ABCDEF12","confidence":0.9,"reasoning":"ok"}'
    # Exact match between pred and label
    eval_pred = EvalPrediction(
        predictions=np.array([pred]),
        label_ids=np.array([pred]),
    )
    metrics = compute_metrics(eval_pred, tok)
    assert metrics["json_validity"] == 1.0
    assert metrics["key_order_match"] == 1.0
    assert metrics["field_type_valid"] == 1.0
    assert metrics["episode_present"] == 1.0
    assert metrics["crc_format"] == 1.0
    assert metrics["exact_json_match"] == 1.0


def test_compute_metrics_partial_failures():
    tok = DummyTokenizer()
    pred = '{"show_name":"A","season":null,"episode":null,"crc_hash":"BAD","confidence":1.2,"reasoning":123}'
    label = '{"show_name":"A","season":1,"episode":2,"crc_hash":"ABCDEF12","confidence":0.9,"reasoning":"ok"}'
    eval_pred = EvalPrediction(
        predictions=np.array([pred]),
        label_ids=np.array([label]),
    )
    metrics = compute_metrics(eval_pred, tok)
    # Valid JSON but types/ranges fail; expect json_validity 1.0 and others less
    assert metrics["json_validity"] == 1.0
    assert metrics["key_order_match"] == 1.0
    assert metrics["field_type_valid"] == 0.0
    assert metrics["episode_present"] == 0.0
    assert metrics["crc_format"] == 0.0
    assert metrics["exact_json_match"] == 0.0

