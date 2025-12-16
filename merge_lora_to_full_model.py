"""
Merge a PEFT LoRA adapter into the base model and save a full merged HF model.

Outputs:
  ./outputs/merged_hf_model/
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main() -> None:
    base_repo = "Qwen/Qwen3-14B"
    adapter_dir = Path("./outputs/sft_model").resolve()
    out_dir = Path("./outputs/merged_hf_model").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prevent accelerate from doing clever offloading that can land weights on "meta"
    os.environ["ACCELERATE_USE_CPU"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dtype = torch.float16  # safe merge dtype; bf16 also fine on 4090
    print(f"[INFO] Loading base: {base_repo}")
    print(f"[INFO] Loading adapter: {adapter_dir}")
    print(f"[INFO] Saving merged model to: {out_dir}")

    tok = AutoTokenizer.from_pretrained(base_repo, trust_remote_code=True)

    # IMPORTANT: do NOT pass device_map="auto"
    base = AutoModelForCausalLM.from_pretrained(
        base_repo,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
    ).to("cuda")

    peft = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = peft.merge_and_unload()

    merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="4GB")
    tok.save_pretrained(str(out_dir))

    print("[OK] Merged HF model written.")


if __name__ == "__main__":
    main()
