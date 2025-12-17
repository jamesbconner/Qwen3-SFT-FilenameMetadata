"""
Learning rate finder (stub).

This stub prints guidance for running an LR sweep. For a real run, replace the
NotImplementedError with a small training loop over a tiny subset and log
loss vs LR.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:  # pragma: no cover
    cfg_path = Path("outputs/sft_model/run_metadata.json")
    cfg_info = {}
    if cfg_path.exists():
        cfg_info = json.loads(cfg_path.read_text(encoding="utf-8"))
    print("Learning Rate Finder (stub)")
    if cfg_info:
        print("Current run metadata:")
        print(json.dumps(cfg_info, indent=2))
    print("\nSuggested next steps:")
    print("- Sample a small subset of training data (e.g., 200-500 examples).")
    print("- Sweep LR logarithmically (e.g., 5e-5 to 5e-3) over ~100 steps.")
    print("- Plot loss vs LR; pick LR just before loss destabilizes.")
    print("- Common good starting LR for LoRA: 2e-4 (already in use).")
    print("\nThis script is a placeholder; implement a sweep if needed.")


if __name__ == "__main__":  # pragma: no cover
    main()

