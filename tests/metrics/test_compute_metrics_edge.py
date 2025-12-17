import numpy as np

import sft_qwen3_metadata as sft


class DummyTok:
    pad_token_id = 0

    def batch_decode(self, seqs, skip_special_tokens=True):
        # Convert to list of strings if not already
        out = []
        for s in seqs:
            if isinstance(s, str):
                out.append(s)
            else:
                out.append("")
        return out


def test_compute_metrics_empty_predictions():
    tok = DummyTok()
    eval_pred = type(
        "EP",
        (),
        {
            "predictions": np.array([""]),
            "label_ids": np.array([""]),
        },
    )()
    metrics = sft.compute_metrics(eval_pred, tok)
    # All zero/neutral when nothing decodes to valid JSON
    assert metrics["json_validity"] == 0.0
    assert metrics["key_order_match"] == 0.0
    assert metrics["field_type_valid"] == 0.0


def test_compute_metrics_logits_guard():
    tok = DummyTok()
    eval_pred = type(
        "EP",
        (),
        {
            "predictions": np.zeros((1, 2, 3), dtype=np.float32),
            "label_ids": np.array([[1, 2]]),
        },
    )()
    metrics = sft.compute_metrics(eval_pred, tok)
    assert all(value == 0.0 for value in metrics.values())
