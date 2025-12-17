import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import merge_lora_to_full_model as mlm


def test_merge_invokes_dependencies(monkeypatch, tmp_path):
    # Arrange: run in a temp workspace
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "outputs"
    adapter_dir = out_dir / "sft_model"
    adapter_dir.mkdir(parents=True)

    calls = SimpleNamespace(tok=False, base=False, peft=False, merged_save=False, tok_save=False)

    class TokMock:
        def save_pretrained(self, path, *args, **kwargs):
            calls.tok_save = True
            Path(path).mkdir(parents=True, exist_ok=True)

    class BaseMock:
        def __init__(self):
            self.to_called = False

        def to(self, *args, **kwargs):
            self.to_called = True
            return self

    class MergedMock:
        def save_pretrained(self, path, *args, **kwargs):
            calls.merged_save = True
            Path(path).mkdir(parents=True, exist_ok=True)

    base_mock = BaseMock()
    merged_mock = MergedMock()

    class PeftMock:
        def __init__(self, base):
            self.base = base

        def merge_and_unload(self):
            return merged_mock

    def tok_factory(repo, trust_remote_code):
        calls.tok = True
        return TokMock()

    def base_factory(repo, torch_dtype, low_cpu_mem_usage, trust_remote_code):
        calls.base = True
        return base_mock

    def peft_factory(base, adapter_path):
        calls.peft = True
        # adapter_path should be the resolved adapter directory
        assert adapter_dir in Path(adapter_path).parents or Path(adapter_path) == adapter_dir
        return PeftMock(base)

    monkeypatch.setattr(mlm, "AutoTokenizer", SimpleNamespace(from_pretrained=tok_factory))
    monkeypatch.setattr(mlm, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=base_factory))
    monkeypatch.setattr(mlm, "PeftModel", SimpleNamespace(from_pretrained=peft_factory))

    # Act
    mlm.main()

    # Assert key calls
    assert calls.tok, "Tokenizer was not loaded"
    assert calls.base, "Base model was not loaded"
    assert calls.peft, "PeftModel was not loaded"
    assert calls.merged_save, "Merged model not saved"
    assert calls.tok_save, "Tokenizer not saved"

    # Env flags set
    assert os.environ.get("ACCELERATE_USE_CPU") == "0"
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"

    # Output dir exists
    assert (tmp_path / "outputs" / "merged_hf_model").exists()

