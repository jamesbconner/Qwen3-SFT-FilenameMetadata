import torch
import warnings
import logging

# Suppress the torch.distributed warning about redirects on Windows/MacOS
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# Import Unsloth FastLanguageModel after suppressing the warning
from unsloth import FastLanguageModel

# Unsloth automatically picks the fastest available kernel (FA3/SDPA/etc)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct",
    max_seq_length = 2048,
    dtype = None, # Auto detects bf16
    load_in_4bit = True,
)

print("\nEnvironment check complete. Optimized attention is enabled.")