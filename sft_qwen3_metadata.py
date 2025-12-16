"""
SFT / QLoRA training for filename -> JSON metadata extraction using Unsloth + Qwen3 14B (4-bit).

This script trains a Qwen3 14B model (via Unsloth) to extract structured metadata from filenames.
The model learns to parse filenames and output JSON with show_name, season, episode, etc.

Training Pipeline:
1. Load CSV with pandas (handles messy data, embedded commas, quotes)
2. Clean and validate data (type coercion, null handling, schema validation)
3. Format examples into Qwen3 chat template (system/user/assistant)
4. Load Qwen3-14B in 4-bit quantization (saves memory)
5. Add LoRA adapters (efficient fine-tuning, only trains small adapter layers)
6. Train using SFTTrainer with packing (multiple examples per sequence)
7. Save LoRA adapters + tokenizer

Expected CSV schema:
name, show_name, season, episode, reasoning, confidence, crc_hash

Key Design Decisions:
- pandas for data cleaning: handles messy CSVs better than raw Python
- 4-bit quantization: allows 14B model to fit on consumer GPUs
- LoRA: only trains ~0.1% of parameters, much faster than full fine-tuning
- Packing: combines multiple short examples into one sequence (better GPU utilization)
- SDPA attention: PyTorch's optimized attention implementation (faster than flash-attn on Windows)
"""

import warnings
import logging

# Suppress Windows-specific warnings from PyTorch distributed training
# (not relevant for single-GPU training)
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# IMPORTANT: import unsloth FIRST so it can patch transformers/trl/peft properly.
# Unsloth monkey-patches these libraries to enable faster training paths.
import unsloth  # noqa: F401

import os
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from multiprocessing import freeze_support

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel, is_bfloat16_supported


# -----------------------
# Configuration
# -----------------------
@dataclass
class CFG:
    """
    Centralized configuration for the training script.
    All hyperparameters and paths are defined here for easy modification.
    """
    # Data paths
    csv_path: str = "./inputs/filename_metadata.csv"  # Input CSV with filename -> metadata mappings
    prompt_dir: str = "./prompts"  # Directory containing prompt templates
    prompt_system: str = "system_filename_sft.txt"  # System prompt (defines task)
    prompt_user: str = "user_filename_sft.txt"  # User prompt template (must contain {filename})

    # Save a cleaned copy for auditing/debugging
    # Useful to verify data cleaning worked correctly
    write_clean_csv: bool = True
    clean_csv_path: str = "./outputs/cleaned_filename_metadata.csv"

    # Model configuration
    model_name: str = "unsloth/Qwen3-14B-bnb-4bit"  # Pre-quantized 4-bit model from Unsloth
    max_seq_length: int = 2048  # Maximum sequence length (affects memory usage)

    # Output directory for saved model
    output_dir: str = "./outputs/sft_model"

    # Reproducibility
    seed: int = 14789  # Random seed for all operations (data split, model init, training)

    # Train/Eval split
    eval_fraction: float = 0.1  # 10% of data for validation

    # Training hyperparameters
    per_device_train_batch_size: int = 1  # Batch size per GPU (1 is safe for 14B model)
    gradient_accumulation_steps: int = 8  # Effective batch size = 1 * 8 = 8
    learning_rate: float = 2e-4  # Learning rate (2e-4 is standard for LoRA)
    num_train_epochs: float = 3.0  # Number of training epochs
    warmup_ratio: float = 0.03  # 3% of steps for warmup (gradual LR increase)
    weight_decay: float = 0.0  # L2 regularization (0.0 = no regularization)

    # Logging/checkpointing
    logging_steps: int = 10  # Log training metrics every N steps
    save_steps: int = 200  # Save checkpoint every N steps
    eval_steps: int = 200  # Run evaluation every N steps
    save_total_limit: int = 2  # Keep only last 2 checkpoints (saves disk space)

    # Performance optimizations
    packing: bool = True  # Pack multiple examples into one sequence (better GPU utilization)
    attn_implementation: str = "sdpa"  # Use PyTorch SDPA (works well on Windows, no flash-attn needed)

    # Data filtering
    require_episode: bool = True  # Drop rows without a parseable episode integer


cfg = CFG()


# -----------------------
# Prompt loading
# -----------------------
def load_prompts(prompt_dir: str, prompt_system: str, prompt_user: str) -> Tuple[str, str]:
    """
    Load prompt templates from files.
    
    The system prompt defines the task (e.g., "Extract metadata from filenames").
    The user prompt template contains {filename} placeholder that gets filled with actual filenames.
    
    Returns:
        Tuple of (system_prompt, user_template_string)
    """
    p = Path(prompt_dir)
    system_path = p / prompt_system
    user_path = p / prompt_user

    if not system_path.exists():
        raise FileNotFoundError(f"Missing system prompt file: {system_path}")
    if not user_path.exists():
        raise FileNotFoundError(f"Missing user prompt file: {user_path}")

    system = system_path.read_text(encoding="utf-8").strip()
    user_tpl = user_path.read_text(encoding="utf-8").strip()

    # Validate that user template has the required placeholder
    if "{filename}" not in user_tpl:
        raise ValueError(f"User prompt template must contain '{{filename}}': {user_path}")

    print(f"[INFO] Using system prompt: {system_path}")
    print(f"[INFO] Using user prompt:   {user_path}")
    return system, user_tpl


SYSTEM_PROMPT, USER_TEMPLATE = load_prompts(cfg.prompt_dir, cfg.prompt_system, cfg.prompt_user)


# -----------------------
# Pandas cleaning helpers
# -----------------------
# Expected columns in the input CSV (validated during loading)
EXPECTED_COLS = ["name", "show_name", "season", "episode", "reasoning", "confidence", "crc_hash"]


def _strip_outer_quotes(s: str) -> str:
    """
    Remove outer quotes from strings (handles both single and double quotes).
    
    Example: '"hello"' -> 'hello', "'world'" -> 'world'
    This is needed because CSV readers sometimes preserve quotes.
    """
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1].strip()
    return s


def _clean_text_series(x: pd.Series) -> pd.Series:
    """
    Clean a text column: convert to string, strip whitespace, remove quotes, normalize nulls.
    
    This handles common CSV issues:
    - Embedded quotes: '"value"' -> 'value'
    - Various null representations: 'nan', 'None', 'NULL' -> pd.NA
    - Whitespace: '  value  ' -> 'value'
    """
    y = x.astype("string")  # Convert to pandas nullable string type
    y = y.fillna(pd.NA)  # Normalize None/NaN to pandas NA
    y = y.str.strip()  # Remove leading/trailing whitespace
    y = y.map(lambda v: _strip_outer_quotes(v) if isinstance(v, str) else v)  # Remove quotes
    # Normalize common null string representations to pandas NA
    y = y.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "NULL": pd.NA})
    return y


def _to_nullable_int(series: pd.Series) -> pd.Series:
    """
    Coerce a series to pandas nullable Int64 type.
    
    Handles various input formats:
    - String integers: '02', '2' -> 2
    - Float integers: 2.0 -> 2
    - String floats: '2.0' -> 2 (only if exactly integral)
    - Non-integral floats: 2.5 -> pd.NA (rejected)
    - Nulls: None, 'nan' -> pd.NA
    
    Returns:
        Series with dtype Int64 (pandas nullable integer type)
    """
    s = series.copy()

    # Clean strings first (handles quotes, whitespace, null strings)
    if s.dtype.name in ("object", "string"):
        s = _clean_text_series(s)

    # Convert to numeric (handles '2.0', '2', etc.)
    num = pd.to_numeric(s, errors="coerce")

    # Keep only values that are very close to an integer (within 1e-9 tolerance)
    # This rejects 2.5 but accepts 2.0, 2.0000001, etc.
    mask_intlike = num.notna() & (np.isclose(num, np.round(num), atol=1e-9))
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")  # Initialize with all NA
    out[mask_intlike] = np.round(num[mask_intlike]).astype(np.int64)  # Fill valid integers
    return out


def _to_float(series: pd.Series, default: float = 0.0) -> pd.Series:
    """
    Coerce a series to float, with default value for nulls.
    
    Used for confidence scores (0.0 to 1.0 range).
    Clips values outside [0.0, 1.0] to stay in valid range.
    """
    s = series.copy()
    if s.dtype.name in ("object", "string"):
        s = _clean_text_series(s)
    out = pd.to_numeric(s, errors="coerce")  # Convert to float, invalid -> NaN
    out = out.fillna(default).astype(float)  # Fill NaN with default (0.0)
    out = out.clip(lower=0.0, upper=1.0)  # Clamp to [0.0, 1.0] range
    return out


def load_and_clean_csv(path: str) -> pd.DataFrame:
    """
    Load and clean CSV file using pandas.
    
    This function handles messy CSVs with:
    - Embedded commas in quoted fields
    - Inconsistent quoting
    - Mixed data types (strings that should be integers)
    - Various null representations
    
    Steps:
    1. Load CSV with Python engine (more forgiving than C engine)
    2. Normalize column names (lowercase, strip whitespace)
    3. Validate schema (check for required columns)
    4. Clean text columns (strip, remove quotes, normalize nulls)
    5. Normalize CRC hashes (uppercase, remove brackets)
    6. Coerce types (season/episode -> Int64, confidence -> float)
    
    Returns:
        Cleaned DataFrame with proper types and normalized values
    """
    # Use Python engine for messy CSVs (handles embedded commas + quotes better than C engine)
    # Read everything as string initially, we'll coerce types later
    df = pd.read_csv(
        path,
        engine="python",
        dtype="string",
        keep_default_na=False,  # Don't auto-convert strings to NaN (we'll handle nulls ourselves)
    )

    # Normalize header names: strip whitespace and convert to lowercase
    # This makes the CSV case-insensitive and whitespace-tolerant
    df.columns = [c.strip().lower() for c in df.columns]

    # Basic schema validation: check for required columns
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLS]

    print(f"[INFO] Raw CSV columns: {list(df.columns)}")
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    if extra:
        print(f"[WARNING] CSV has extra columns (will be ignored): {extra}")

    # Keep only expected columns in canonical order
    df = df[EXPECTED_COLS].copy()

    # Clean text columns: strip, remove quotes, normalize nulls
    for col in ["name", "show_name", "reasoning", "crc_hash"]:
        df[col] = _clean_text_series(df[col])

    # Special handling for CRC hash: normalize format
    # Handles cases like "[A1B2C3D4]" -> "A1B2C3D4" and converts to uppercase
    df["crc_hash"] = df["crc_hash"].fillna(pd.NA)
    df["crc_hash"] = df["crc_hash"].astype("string").str.strip()
    df["crc_hash"] = df["crc_hash"].str.replace(r"^\[([A-Fa-f0-9]{8})\]$", r"\1", regex=True)  # Remove brackets
    df["crc_hash"] = df["crc_hash"].str.upper()  # Convert to uppercase
    df.loc[df["crc_hash"] == "", "crc_hash"] = pd.NA  # Empty strings -> NA

    # Coerce numeric types
    df["season"] = _to_nullable_int(df["season"])  # Int64 (nullable integer)
    df["episode"] = _to_nullable_int(df["episode"])  # Int64 (nullable integer)
    df["confidence"] = _to_float(df["confidence"], default=0.0)  # float [0.0, 1.0]

    return df


def print_descriptive_stats(df: pd.DataFrame) -> None:
    """
    Print comprehensive statistics about the cleaned dataset.
    
    This helps verify data quality and understand the distribution:
    - Total row count
    - Null value counts per column
    - Episode number distribution (min/max/mean, most common values)
    - Season coverage percentage
    - Confidence score distribution
    - Sample rows for manual inspection
    """
    print("\n" + "=" * 70)
    print("Dataset sanity checks (pandas)")
    print("=" * 70)
    print(f"[INFO] rows: {len(df)}")
    print("[INFO] null counts:")
    print(df.isna().sum().to_string())

    # Episode distribution: show stats for non-null episodes
    # This helps identify if episodes are in a reasonable range (e.g., 1-100)
    ep = df["episode"]
    print("\n[INFO] episode stats (non-null):")
    if ep.notna().any():
        print(ep.dropna().astype(int).describe().to_string())  # min, max, mean, etc.
        print("\n[INFO] top 20 episode values:")
        print(ep.dropna().astype(int).value_counts().head(20).to_string())  # Most common episodes
    else:
        print("[INFO] no non-null episodes found")

    # Season coverage: what percentage of rows have season info
    se = df["season"]
    season_pct = float(se.notna().mean()) * 100.0
    print(f"\n[INFO] season present: {season_pct:.2f}%")

    # Confidence distribution: check if confidence scores are reasonable
    print("\n[INFO] confidence stats:")
    print(df["confidence"].describe().to_string())

    # Quick peek at a few rows for manual inspection
    # Helps catch data quality issues early
    print("\n[INFO] sample rows:")
    sample = df.sample(min(5, len(df)), random_state=cfg.seed)
    print(sample.to_string(index=False))


# -----------------------
# JSON target formatting
# -----------------------
def row_to_target_json(row: Dict[str, Any]) -> str:
    """
    Convert a data row to JSON string that the model should output.
    
    This is the target format the model learns to generate.
    The JSON has a fixed key order for consistency.
    
    Args:
        row: Dictionary with keys: show_name, season, episode, crc_hash, confidence, reasoning
        
    Returns:
        Compact JSON string (no extra whitespace, ASCII-safe)
        
    Example output:
        {"show_name":"The Office","season":2,"episode":5,"crc_hash":"A1B2C3D4","confidence":0.95,"reasoning":"..."}
    """
    obj = {
        "show_name": row.get("show_name") or "",  # Empty string if None
        "season": row.get("season"),  # Can be None (nullable)
        "episode": row.get("episode"),  # Can be None (nullable)
        "crc_hash": row.get("crc_hash"),  # Can be None (nullable)
        "confidence": float(row.get("confidence") or 0.0),  # Always a float
        "reasoning": row.get("reasoning") or "",  # Empty string if None
    }
    # Compact JSON: no extra spaces, ASCII-safe (handles Unicode in strings)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def format_example(row: Dict[str, Any]) -> Dict[str, str]:
    """
    Format a data row into Qwen3 chat template format.
    
    Qwen3 uses a specific chat format with special tokens:
    - <|system|>: System prompt (defines the task)
    - <|user|>: User input (the filename to parse)
    - <|assistant|>: Expected output (JSON metadata)
    
    This format is what the model was trained on, so we must match it exactly.
    
    Args:
        row: Dictionary with CSV row data
        
    Returns:
        Dictionary with single "text" key containing the formatted conversation
    """
    filename = row.get("name") or ""  # Get filename from row
    user = USER_TEMPLATE.format(filename=filename)  # Fill in {filename} placeholder
    assistant = row_to_target_json(row)  # Generate target JSON output
    
    # Format as Qwen3 chat template (matches the model's training format)
    text = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n{assistant}"
    )
    return {"text": text}


# -----------------------
# LoRA target module discovery
# -----------------------
def find_lora_target_modules(model) -> List[str]:
    """
    Automatically discover which linear layers to apply LoRA adapters to.
    
    LoRA (Low-Rank Adaptation) only trains small adapter matrices on top of existing weights.
    We target attention and MLP layers:
    - Attention: q_proj, k_proj, v_proj (query/key/value), o_proj (output)
    - MLP: gate_proj, up_proj, down_proj (feed-forward network)
    
    Different model architectures use different naming conventions, so we try:
    1. Standard naming (q_proj, k_proj, etc.) - used by LLaMA, Qwen, etc.
    2. Alternative naming (Wqkv, Wq, etc.) - used by some other architectures
    
    Args:
        model: The loaded model (after from_pretrained)
        
    Returns:
        List of module names to apply LoRA to (e.g., ["q_proj", "k_proj", "v_proj", ...])
    """
    # Standard naming convention (LLaMA, Qwen, Mistral, etc.)
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = set()

    # These are the linear layer types we're looking for
    # Linear4bit = 4-bit quantized linear layer (from bitsandbytes)
    linear_names = {"Linear", "Linear4bit", "Linear8bitLt"}
    
    # Scan all modules in the model
    for name, module in model.named_modules():
        if module.__class__.__name__ not in linear_names:
            continue
        # Get the leaf name (last part of the path, e.g., "q_proj" from "model.layers.0.self_attn.q_proj")
        leaf = name.split(".")[-1]
        if leaf in wanted:
            found.add(leaf)

    # Return in canonical order (important for reproducibility)
    ordered = [m for m in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if m in found]
    if ordered:
        return ordered

    # Fallback: try alternative naming conventions (some models use different names)
    alt = {"Wqkv", "Wq", "Wk", "Wv", "Wo", "wqkv", "wo"}
    found2 = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ not in linear_names:
            continue
        leaf = name.split(".")[-1]
        if leaf in alt:
            found2.add(leaf)

    return [m for m in ["Wqkv", "Wq", "Wk", "Wv", "Wo", "wqkv", "wo"] if m in found2]


# -----------------------
# Main training pipeline
# -----------------------
def main() -> None:
    """
    Main training function that orchestrates the entire pipeline.
    
    Pipeline steps:
    1. Set random seeds for reproducibility
    2. Load and clean CSV data
    3. Filter data (optional: require episode)
    4. Convert to HuggingFace Dataset format
    5. Format examples into Qwen3 chat template
    6. Load pre-quantized model (4-bit)
    7. Add LoRA adapters
    8. Configure training arguments
    9. Train the model
    10. Save LoRA adapters and tokenizer
    """
    freeze_support()  # Required for Windows multiprocessing

    # Set random seeds for reproducibility
    # This ensures the same data split, model initialization, and training behavior
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # ============================================
    # Step 1: Load and clean CSV via pandas
    # ============================================
    # pandas handles messy CSVs better than raw Python (embedded commas, quotes, etc.)
    df = load_and_clean_csv(cfg.csv_path)
    print_descriptive_stats(df)  # Print stats to verify data quality

    # ============================================
    # Step 2: Optional data filtering
    # ============================================
    # Filter out rows without episode numbers (if required)
    # This ensures we only train on examples with complete metadata
    if cfg.require_episode:
        before = len(df)
        df = df[df["episode"].notna()].copy()  # Keep only rows with non-null episode
        after = len(df)
        print(f"\n[INFO] require_episode=True: filtered rows {before} -> {after}")
        if after == 0:
            raise RuntimeError(
                "After cleaning, there are 0 rows with a non-null episode. "
                "This means the CSV still does not contain episode values in a parseable form."
            )

    # ============================================
    # Step 3: Write cleaned CSV for debugging
    # ============================================
    # Save a cleaned version for manual inspection and debugging
    if cfg.write_clean_csv:
        Path(cfg.clean_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.clean_csv_path, index=False, encoding="utf-8")
        print(f"[INFO] Wrote cleaned CSV: {cfg.clean_csv_path}")

    # ============================================
    # Step 4: Build HuggingFace Dataset from pandas
    # ============================================
    # Convert pandas DataFrame to HuggingFace Dataset format
    # HuggingFace Dataset is required by the training framework
    
    # Ensure nullable integer types are preserved
    df_hf = df.copy()
    df_hf["season"] = df_hf["season"].astype("Int64")
    df_hf["episode"] = df_hf["episode"].astype("Int64")

    # Convert DataFrame to list of dictionaries (one dict per row)
    records = df_hf.to_dict(orient="records")
    dataset = Dataset.from_list(records)  # Create HuggingFace Dataset

    # Split into train/eval sets (90/10 split by default)
    split = dataset.train_test_split(test_size=cfg.eval_fraction, seed=cfg.seed)
    
    # Format each example into Qwen3 chat template
    # On Windows, use num_proc=1 to avoid multiprocessing issues
    train_ds = split["train"].map(format_example, num_proc=1 if os.name == "nt" else None)
    eval_ds = split["test"].map(format_example, num_proc=1 if os.name == "nt" else None)

    # Remove all columns except "text" (SFTTrainer only needs the formatted text)
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c != "text"])
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c != "text"])

    # ============================================
    # Step 5: Load model and tokenizer
    # ============================================
    # Choose dtype based on GPU support (bfloat16 is better if available)
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16

    # Load pre-quantized 4-bit model from Unsloth
    # This model is already quantized, so it uses much less memory than full precision
    # max_seq_length: maximum sequence length the model can handle
    # load_in_4bit: use 4-bit quantization (already done in this model, but flag is required)
    # attn_implementation: use SDPA (works well on Windows, no flash-attn needed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
        attn_implementation=cfg.attn_implementation,
    )

    # Automatically discover which layers to apply LoRA to
    # LoRA only trains small adapter matrices, not the full model weights
    target_modules = find_lora_target_modules(model)
    print(f"[INFO] LoRA target_modules = {target_modules}")
    if not target_modules:
        raise RuntimeError("Could not discover LoRA target modules. Inspect model.named_modules().")

    # ============================================
    # Step 6: Add LoRA adapters
    # ============================================
    # LoRA (Low-Rank Adaptation) adds small trainable matrices on top of existing weights
    # This allows fine-tuning with ~0.1% of the original parameters
    # 
    # r=16: rank of the LoRA matrices (higher = more capacity, but more parameters)
    # lora_alpha=16: scaling factor (typically equals r)
    # lora_dropout=0.0: no dropout (Unsloth's fast path, slightly faster training)
    # use_gradient_checkpointing="unsloth": saves memory by recomputing activations
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank (controls adapter size)
        target_modules=target_modules,  # Which layers to apply LoRA to
        lora_alpha=16,  # LoRA alpha (scaling factor)
        lora_dropout=0.0,  # No dropout (Unsloth fast path)
        bias="none",  # Don't train bias terms (saves memory)
        use_gradient_checkpointing="unsloth",  # Memory-efficient training
        random_state=cfg.seed,  # Reproducible LoRA initialization
        max_seq_length=cfg.max_seq_length,
    )

    # ============================================
    # Step 7: Configure training arguments
    # ============================================
    # TrainingArguments controls all aspects of training (optimizer, learning rate, etc.)
    args = TrainingArguments(
        output_dir=cfg.output_dir,  # Where to save checkpoints
        seed=cfg.seed,  # Random seed for training

        # Batch configuration
        # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
        per_device_train_batch_size=cfg.per_device_train_batch_size,  # 1 example per GPU
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,  # Accumulate over 8 steps
        # So effective batch size = 1 * 8 = 8

        # Learning rate and schedule
        learning_rate=cfg.learning_rate,  # 2e-4 is standard for LoRA
        num_train_epochs=cfg.num_train_epochs,  # Train for 3 epochs
        warmup_ratio=cfg.warmup_ratio,  # 3% of steps for warmup (gradual LR increase)
        weight_decay=cfg.weight_decay,  # L2 regularization (0.0 = disabled)

        # Logging and checkpointing
        logging_steps=cfg.logging_steps,  # Log every 10 steps
        save_steps=cfg.save_steps,  # Save checkpoint every 200 steps
        eval_strategy="steps",  # Evaluate during training (not just at end)
        eval_steps=cfg.eval_steps,  # Evaluate every 200 steps
        save_total_limit=cfg.save_total_limit,  # Keep only last 2 checkpoints

        # Mixed precision training (reduces memory usage)
        bf16=(dtype == torch.bfloat16),  # Use bfloat16 if supported
        fp16=(dtype == torch.float16),  # Use float16 otherwise

        # Optimizer and scheduler
        optim="adamw_torch",  # AdamW optimizer (standard for transformers)
        lr_scheduler_type="cosine",  # Cosine learning rate schedule

        report_to="none",  # Don't report to external services (W&B, etc.)
    )

    # ============================================
    # Step 8: Create trainer and train
    # ============================================
    # SFTTrainer (Supervised Fine-Tuning Trainer) handles the training loop
    # It formats data, computes loss, updates weights, etc.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,  # Training examples
        eval_dataset=eval_ds,  # Validation examples
        dataset_text_field="text",  # Column name containing formatted text
        max_seq_length=cfg.max_seq_length,  # Maximum sequence length
        packing=cfg.packing,  # Pack multiple examples into one sequence (better GPU utilization)
        args=args,  # Training arguments
    )

    # ============================================
    # Step 9: Train the model
    # ============================================
    # This is where the actual training happens
    # The trainer will:
    # - Iterate through training examples
    # - Compute loss (how far off the model's predictions are)
    # - Update LoRA adapter weights via backpropagation
    # - Periodically evaluate on validation set
    # - Save checkpoints
    t0 = time.time()
    trainer.train()  # Start training (this can take a while!)
    t1 = time.time()

    # ============================================
    # Step 10: Save the trained model
    # ============================================
    # Save LoRA adapters (the trained weights) and tokenizer
    # Note: We only save the adapters, not the full model (saves disk space)
    # To use the model later, load the base model + these adapters
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"\nTraining complete in {(t1 - t0) / 60.0:.2f} minutes")
    print(f"Saved adapters + tokenizer to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
