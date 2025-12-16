# LLM Fine-Tuning: Filename Metadata Extraction

A complete pipeline for fine-tuning Qwen3 14B to extract structured metadata from filenames using Supervised Fine-Tuning (SFT) with QLoRA. The model learns to parse filenames and output JSON with show names, seasons, episodes, CRC hashes, confidence scores, and reasoning.

## 🎯 Project Overview

This project trains a large language model (Qwen3 14B) to extract structured metadata from filenames. The model takes a filename as input and outputs a JSON object containing:
- `show_name`: Name of the TV show or series
- `season`: Season number (nullable integer)
- `episode`: Episode number (nullable integer)
- `crc_hash`: CRC32 hash of the file (nullable string)
- `confidence`: Confidence score (0.0 to 1.0)
- `reasoning`: Explanation of the extraction

### Key Features

- **Efficient Training**: Uses QLoRA (4-bit quantization + LoRA adapters) to fine-tune a 14B parameter model on consumer GPUs
- **Robust Data Processing**: Pandas-based CSV cleaning handles messy data, embedded commas, and inconsistent formatting
- **Optimized for Windows**: Uses SDPA attention (works without flash-attn compilation issues)
- **Fast Training**: Leverages Unsloth for 2x faster training with optimized kernels
- **Ollama Integration**: Includes scripts to convert trained model to GGUF format for local inference via Ollama

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4090)
- **VRAM**: ~24GB+ recommended for 14B model in 4-bit
- **RAM**: 16GB+ system RAM

### Software
- **OS**: Windows 10/11 (Linux/macOS should work with minor modifications)
- **Python**: 3.11+ (tested with 3.11.14)
- **CUDA**: 12.8+ toolkit installed
- **Git**: For version control

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LLM_Fine_Tuning
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/macOS
```

### 3. Install Dependencies

See [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions.

**Quick install:**
```bash
# Install PyTorch with CUDA 12.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies
pip install transformers>=4.51.0 datasets peft accelerate bitsandbytes trl ninja packaging

# Install Unsloth (see INSTALLATION.md for Windows-specific installation method)
pip install "unsloth[cu12]"

# Verify xformers installation
python -m xformers.info
```

### 4. Verify Installation

```bash
python scripts/verify_cuda.py
```

This will check:
- PyTorch and CUDA availability
- Attention backends (xformers, SDPA, Triton)
- Unsloth installation

## 📁 Project Structure

```
LLM_Fine_Tuning/
├── inputs/                    # Input data (CSV files) - gitignored
│   └── filename_metadata.csv
├── outputs/                   # Training outputs - gitignored
│   ├── sft_model/           # LoRA adapters
│   ├── merged_hf_model/     # Merged full model
│   └── *.gguf              # Quantized models for Ollama
├── prompts/                  # Prompt templates
│   ├── system_filename_sft.txt
│   └── user_filename_sft.txt
├── scripts/                  # Utility scripts
│   ├── verify_cuda.py      # Environment verification
│   └── verify_unsloth.py
├── tests/                    # Test fixtures
│   └── fixtures/
│       └── validation_filenames_hard.txt
├── sft_qwen3_metadata.py    # Main training script
├── infer_validate.py        # Inference and validation
├── merge_lora_to_full_model.py  # Merge LoRA adapters to base model
├── Modelfile                # Ollama model configuration
├── INSTALLATION.md          # Detailed installation guide
├── PROCESS.md               # Training workflow
└── README.md               # This file
```

## 🔧 Usage

> **Note**: For a complete step-by-step workflow, see [PROCESS.md](PROCESS.md).

### Training the Model

1. **Prepare your data**: Place a CSV file in `inputs/` with the following columns:
   - `name`: Filename
   - `show_name`: TV show name
   - `season`: Season number (integer or null)
   - `episode`: Episode number (integer or null)
   - `reasoning`: Extraction reasoning
   - `confidence`: Confidence score (0.0 to 1.0)
   - `crc_hash`: CRC32 hash (optional)

2. **Configure prompts**: Edit `prompts/system_filename_sft.txt` and `prompts/user_filename_sft.txt` to match your task.

3. **Run training**:
   ```bash
   python sft_qwen3_metadata.py
   ```
   
   Training takes approximately 40 minutes on an RTX 4090.

4. **Validate the model**:
   ```bash
   python infer_validate.py
   ```

### Converting to Ollama Format

To use the model with Ollama for local inference:

1. **Merge LoRA adapters**: This merges the trained LoRA adapters into the full-precision base model (`Qwen/Qwen3-14B`). The merged model is saved to `outputs/merged_hf_model/`.
   ```bash
   python merge_lora_to_full_model.py
   ```
   
   **Note**: This requires loading the full 14B model in memory (~28GB RAM/VRAM). The script loads `Qwen/Qwen3-14B` (not the quantized version) and merges the LoRA adapters into it.

2. **Download llama.cpp**: Get the latest release from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)

3. **Convert to GGUF** (requires llama.cpp):
   ```bash
   python <path-to-llama.cpp>/convert_hf_to_gguf.py outputs/merged_hf_model \
       --outfile outputs/qwen3-animemetadata-f16.gguf \
       --outtype f16
   ```

4. **Quantize** (optional, for smaller model):
   ```bash
   <path-to-llama.cpp-binaries>/llama-quantize.exe outputs/qwen3-animemetadata-f16.gguf \
       outputs/qwen3-animemetadata-q4_k_m.gguf Q4_K_M
   ```

5. **Update Modelfile**: Ensure `Modelfile` points to the correct GGUF path (default: `./outputs/qwen3-animemetadata-q4_k_m.gguf`)

6. **Create Ollama model**:
   ```bash
   ollama create qwen3-animemetadata -f Modelfile
   ```

## ⚙️ Configuration

Training hyperparameters are configured in the `CFG` dataclass in `sft_qwen3_metadata.py`:

```python
@dataclass
class CFG:
    # Model
    model_name: str = "unsloth/Qwen3-14B-bnb-4bit"
    max_seq_length: int = 2048
    
    # Training
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    
    # Data
    csv_path: str = "./inputs/filename_metadata.csv"
    eval_fraction: float = 0.1
    require_episode: bool = True
```

Modify these values to adjust training behavior.

## 🧪 Testing

Run validation on difficult test cases:

```bash
python infer_validate.py
```

This script tests the model on hardcoded edge cases (difficult filenames with various formatting challenges). The test cases include filenames with Unicode characters, embedded metadata, and complex naming patterns.

## 📊 Training Details

### Model Architecture
- **Base Model**: Qwen3-14B (loaded via Unsloth's pre-quantized 4-bit version for training)
- **Training Quantization**: 4-bit (bitsandbytes) - allows 14B model to fit on consumer GPUs
- **Fine-tuning Method**: QLoRA (LoRA rank=16, alpha=16, dropout=0.0)
- **Attention**: SDPA (PyTorch's optimized implementation - works on Windows without flash-attn)
- **Merged Model**: Full precision Qwen3-14B with LoRA adapters merged (for inference)

### Training Configuration
- **Effective Batch Size**: 8 (1 per device × 8 gradient accumulation steps)
- **Learning Rate**: 2e-4 with cosine schedule
- **Epochs**: 3.0
- **Sequence Packing**: Enabled (better GPU utilization)
- **Mixed Precision**: bfloat16 (if supported) or float16

### Performance
- **Training Time**: ~40 minutes on RTX 4090
- **Memory Usage**: ~20-24GB VRAM
- **Model Size**: 
  - LoRA adapters: ~50MB
  - Merged model: ~8GB (FP16) or ~4GB (Q4_K_M quantized)

## 🐛 Troubleshooting

### CUDA Issues
- **CUDA_HOME not found**: Ensure CUDA 12.8 is installed and `CUDA_HOME` environment variable is set
- **CUDA version mismatch**: PyTorch CUDA version must match CUDA toolkit version
- **Out of memory**: Reduce `max_seq_length` or `per_device_train_batch_size`

### Installation Issues
- **flash-attn fails to build**: Not needed! The project uses SDPA which works without flash-attn
- **xformers issues**: Verify installation with `python -m xformers.info`
- **Unsloth import errors**: Ensure Unsloth is imported before other transformers libraries

### Training Issues
- **No LoRA modules found**: Check model architecture compatibility. The script auto-discovers target modules (q_proj, k_proj, v_proj, etc.)
- **Data loading errors**: Verify CSV schema matches expected columns (`name`, `show_name`, `season`, `episode`, `reasoning`, `confidence`, `crc_hash`)
- **Validation fails**: Check that prompts are correctly formatted and contain `{filename}` placeholder in user prompt
- **Episode filtering removes all data**: Set `require_episode=False` in config if your data doesn't have episode numbers

## 📚 Documentation

- [INSTALLATION.md](INSTALLATION.md) - Detailed installation instructions
- [PROCESS.md](PROCESS.md) - Complete training and deployment workflow

## 🔍 Verification Scripts

- `scripts/verify_cuda.py`: Comprehensive environment check
  - PyTorch and CUDA status
  - Attention backends (xformers, SDPA, Triton)
  - Unsloth installation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning framework
- [Qwen Team](https://github.com/QwenLM/Qwen) - Qwen3 model
- [Hugging Face](https://huggingface.co/) - Transformers library and model hub
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF conversion tools

## 📧 Contact

[Add your contact information]

---

**Note**: This project is optimized for Windows but should work on Linux/macOS with minor modifications (primarily path separators and multiprocessing settings).

