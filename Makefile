# Make targets for SFT + QLoRA + Ollama pipeline
# Adjust paths/vars below as needed.

# -------- Variables --------
# Use explicit Windows-style paths so cmd /C can find the venv binaries.
PYTHON      ?= .venv\\Scripts\\python.exe
PIP         ?= .venv\\Scripts\\pip.exe
TORCH_INDEX ?= https://download.pytorch.org/whl/cu128

# llama.cpp paths (set these to your local install)
LLAMACPP_CONVERT ?= D:/Languages/llama/llama.cpp-b7415/convert_hf_to_gguf.py
LLAMACPP_QUANT   ?= D:/Languages/llama/llama-b7415-bin-win-cuda-12.4-x64/llama-quantize.exe

# Model/output paths
MERGED_HF_DIR    ?= outputs/merged_hf_model
SFT_OUTPUT_DIR   ?= outputs/sft_model
GGUF_F16         ?= outputs/qwen3-filemetadata-f16.gguf
GGUF_Q4          ?= outputs/qwen3-filemetadata-q4_k_m.gguf
MODEFILE_PATH    ?= Modelfile

# -------- Phony targets --------
.PHONY: help venv install train validate merge gguf quantize ollama all test clean zip clean_outputs bench lr_finder

help:
	@echo "make venv          - create virtual env (.venv)"
	@echo "make install       - install dependencies (torch w/ CUDA 12.8 + libs)"
	@echo "make train         - run SFT QLoRA training"
	@echo "make validate      - run validation script"
	@echo "make merge         - merge LoRA into full HF model"
	@echo "make gguf          - convert merged HF to GGUF (f16)"
	@echo "make quantize      - quantize GGUF to Q4_K_M"
	@echo "make ollama        - create ollama model from Modelfile"
	@echo "make test          - run pytest"
	@echo "make clean         - remove caches/pyc (keeps outputs/, inputs/, .venv/)"
	@echo "make clean_outputs - remove generated outputs/ (destructive)"
	@echo "make zip           - create distribution zip (excludes outputs/, inputs/, .venv/)"
	@echo "make bench         - run a short inference benchmark"
	@echo "make lr_finder     - run LR finder stub/instructions"

venv:
	python -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision --index-url $(TORCH_INDEX)
	$(PIP) install transformers>=4.51.0 datasets peft accelerate bitsandbytes trl ninja packaging pytest-cov ruff mypy
	$(PIP) install "unsloth[cu12]"
	$(PIP) install pytest

train:
	@echo Setting CUDA debug env and running training...
	cmd /C "set CUDA_LAUNCH_BLOCKING=1 && set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && $(PYTHON) sft_qwen3_metadata.py"

validate:
	$(PYTHON) infer_validate.py

merge:
	@echo Skipping torchao and normalizing allocator settings for merge...
	cmd /C "set TRANSFORMERS_NO_TORCHAO=1 && set PYTORCH_CUDA_ALLOC_CONF= && set PYTORCH_ALLOC_CONF=max_split_size_mb:128 && $(PYTHON) merge_lora_to_full_model.py"

gguf:
	$(PYTHON) $(LLAMACPP_CONVERT) $(MERGED_HF_DIR) --outfile $(GGUF_F16) --outtype f16

quantize:
	"$(LLAMACPP_QUANT)" $(GGUF_F16) $(GGUF_Q4) Q4_K_M

ollama:
	ollama create qwen3-filemetadata -f $(MODEFILE_PATH)

test:
	$(PYTHON) -m pytest

bench:
	$(PYTHON) infer_validate.py --benchmark

lr_finder:
	$(PYTHON) lr_finder.py

clean:
	powershell -Command " \
	  Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; \
	  Get-ChildItem -Recurse -Include *.pyc,*.pyo -File | Remove-Item -Force -ErrorAction SilentlyContinue; \
	  if (Test-Path .pytest_cache) { Remove-Item -Recurse -Force .pytest_cache }; \
	  if (Test-Path .cache) { Remove-Item -Recurse -Force .cache } \
	"

clean_outputs:
	powershell -Command " \
	  if (Test-Path outputs) { Remove-Item -Recurse -Force outputs }; \
	  if (Test-Path unsloth_compiled_cache) { Remove-Item -Recurse -Force unsloth_compiled_cache } \
	"

zip:
	@if not exist .git ( \
		echo git repo not found; cannot use git archive & exit /b 1 \
	)
	git archive --format=zip --output=llm_fine_tuning.zip HEAD

