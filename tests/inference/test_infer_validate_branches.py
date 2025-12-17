import logging
import torch
import pytest

import infer_validate as iv


def test_configure_logging_file(tmp_path):
    log_path = tmp_path / "iv.log"
    logger = iv.configure_logging(name="iv_test_logger", log_file=str(log_path))
    logger.info("hello")
    assert log_path.exists()


def test_build_inputs_handles_missing_pad_token():
    class Tok:
        pad_token_id = None

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors):
            assert tokenize and add_generation_prompt and return_tensors == "pt"
            return torch.tensor([[1, 2, 3]])

    out = iv.build_inputs(Tok(), "sys", "user")
    assert out["input_ids"].tolist() == [[1, 2, 3]]
    assert out["attention_mask"].tolist() == [[1, 1, 1]]


def test_load_prompts_missing_and_placeholder(tmp_path):
    sys_p = tmp_path / "sys.txt"
    usr_p = tmp_path / "usr.txt"
    sys_p.write_text("S", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        iv.load_prompts(str(tmp_path), "sys.txt", "usr.txt")
    usr_p.write_text("no placeholder here", encoding="utf-8")
    with pytest.raises(ValueError):
        iv.load_prompts(str(tmp_path), "sys.txt", "usr.txt")


def test_stop_on_balanced_json_short():
    tok = type("Tok", (), {"decode": lambda self, ids, skip_special_tokens=True: ""})()
    stop = iv.StopOnBalancedJSON(tok, prompt_len=1)
    ids = torch.tensor([[1]])  # gen_ids length < 2
    assert stop(ids, None) is False


def test_extract_first_balanced_json_unbalanced():
    obj, err = iv.extract_first_balanced_json("{abc")
    assert obj is None
    assert "unbalanced" in err


def test_generate_with_retry_failure():
    class Tok:
        eos_token_id = 0
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "no json here"

    class Model:
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    inputs = {"input_ids": torch.tensor([[1, 2]])}
    stop = iv.StoppingCriteriaList([iv.StopOnBalancedJSON(Tok(), prompt_len=1)])
    result = iv.generate_with_retry(Model(), Tok(), inputs, stop, max_retries=1)
    assert result["status"] == "fail"
    assert result["errors"]


def test_run_failure_branch(monkeypatch, caplog, tmp_path):
    # Avoid CUDA dependency and .to('cuda') failures
    monkeypatch.setattr(iv.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(iv, "is_bfloat16_supported", lambda: False)

    orig_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)

    class Tok:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, prompt, return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}

        def decode(self, ids, skip_special_tokens=True):
            return "no json here"

    class Model:
        def load_adapter(self, *a, **k):
            return None

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    monkeypatch.setattr(iv.FastLanguageModel, "from_pretrained", lambda *a, **k: (Model(), Tok()))
    monkeypatch.setattr(iv.FastLanguageModel, "for_inference", lambda m: None)

    with caplog.at_level(logging.ERROR):
        iv.run(benchmark=True)
    assert any("[FAIL]" in r.message for r in caplog.records)

    monkeypatch.setattr(torch.Tensor, "to", orig_to, raising=False)

