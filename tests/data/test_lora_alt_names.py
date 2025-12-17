import sft_qwen3_metadata as sft


def test_find_lora_target_modules_alt_names():
    Linear4bit = type("Linear4bit", (), {})

    def mod():
        return Linear4bit()

    class Model:
        def named_modules(self):
            return [
                ("layers.0.attn.Wqkv", mod()),
                ("layers.0.attn.Wo", mod()),
            ]

    found = sft.find_lora_target_modules(Model())
    assert found == ["Wqkv", "Wo"]

