import torch

import infer_validate as iv


def test_main_runs_with_mocks(monkeypatch, caplog, tmp_path):
    # Run in isolated cwd
    monkeypatch.chdir(tmp_path)

    # Reduce test set for speed
    monkeypatch.setattr(iv, "TEST_FILENAMES", ["file1"])
    monkeypatch.setattr(iv, "is_bfloat16_supported", lambda: False)

    # Mock tokenizer and model
    class TokMock:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, prompt, return_tensors=None):
            # Minimal tokenization
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

        def decode(self, ids, skip_special_tokens=True):
            # Always return a valid JSON for both stopping and final decode
            return (
                '{"show_name":"A","season":1,"episode":2,'
                '"crc_hash":"ABCDEF12","confidence":0.9,"reasoning":"ok"}'
            )

    class ModelMock:
        def load_adapter(self, *args, **kwargs):
            self.adapter_loaded = True

        def generate(self, **kwargs):
            # Return prompt + generated ids; prompt length is 3
            return torch.tensor([[1, 2, 3, 9, 9]])

    model = ModelMock()
    tok = TokMock()

    monkeypatch.setattr(iv.FastLanguageModel, "from_pretrained", lambda *a, **k: (model, tok))
    monkeypatch.setattr(iv.FastLanguageModel, "for_inference", lambda m: None)

    with caplog.at_level("INFO"):
        iv.main([])
    assert any("PASS" in rec.message for rec in caplog.records)


def test_stop_on_balanced_json_stops(monkeypatch):
    tok = type(
        "Tok",
        (),
        {
            "decode": lambda self, ids, skip_special_tokens=True: '{"a":1}',
        },
    )()
    stop = iv.StopOnBalancedJSON(tok, prompt_len=1)
    input_ids = torch.tensor([[5, 6, 7]])  # gen part is [6,7]
    assert stop(input_ids, None) is True

