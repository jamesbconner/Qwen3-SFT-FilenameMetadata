# Installation Steps

## Install CUDA 12.8

	* https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
	* Env vars ... CUDA_HOME, CUDA_PATH, PATH
	* nvcc --version

## Install PyTorch

	* https://pytorch.org/get-started/locally/
		- PyTorch Build 2.9.1, Windows, pip, Python, CUDA 12.8
		- pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
	* python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

## Install supporting libraries

	* pip install transformers>=4.51.0 datasets peft accelerate bitsandbytes trl ninja packaging
	* python -m xformers.info

## Install UnSloth

	* https://docs.unsloth.ai/get-started/install-and-update/windows-installation
		- Method #3 - Install directly on Windows
	* Test UnSloth
```python
from unsloth import FastLanguageModel

# Unsloth automatically picks the fastest available kernel (FA3/SDPA/etc)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct",
    max_seq_length = 2048,
    dtype = None, # Auto detects bf16
    load_in_4bit = True,
)
print("\nEnvironment check complete. Optimized attention is enabled.")
```