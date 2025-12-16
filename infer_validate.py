import json
import re
import sys
import time
import warnings
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch

# Suppress the torch.distributed warning about redirects on Windows/MacOS
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# IMPORTANT: import unsloth first
import unsloth  # noqa: F401
from unsloth import FastLanguageModel, is_bfloat16_supported

from transformers import StoppingCriteria, StoppingCriteriaList


# -----------------------
# Config
# -----------------------
BASE_MODEL = "unsloth/Qwen3-14B-bnb-4bit"
ADAPTER_DIR = "./outputs/sft_model"
MAX_SEQ_LEN = 2048
ATTN_IMPL = "sdpa"

PROMPT_DIR = "./prompts"
SYSTEM_FILE = "system_filename_sft.txt"
USER_FILE = "user_filename_sft.txt"

REQUIRE_EPISODE = True

# Make this smaller while debugging so you always see output fast
MAX_NEW_TOKENS = 256

TEST_FILENAMES = [
    "Usagi-senpai wa īsutā no yume o mite iru - 27 (1080p).ADN.WEB-DL.AAC2.0.H.264-VARYG.[23A0FB72].mkv",
    "[Bunny]Burankugēto.-.uchū.no.nazo.Season1_Eps22(720p).FLAC.H.265.[39AB5490].mkv",
    "Isekai_de_tsuri_ni_dekaketa-Midori_Blues_E03_(4k).mkv",
    "Dare ga inu-sama o soto ni dashita nda? - S03 - E07 - 3-Ri no baka ga kataru monogatari 1080p AV1 [687A59F9].mkv",
    "Supēsuopera---aku-no-teiō-sama-to-sutārōdo-kun-kimi-6-gō-to-no-saishū-kessen-02 (S01E02v2).mkv",
    "Matte, kono nioi wa nani? 7-Jigen no mukō kara kita neko-chan no nioida! S02 -11- (AV1) (HVEC) (x265).mkv",
]

EXPECTED_KEYS = ["show_name", "season", "episode", "crc_hash", "confidence", "reasoning"]
CRC_RE = re.compile(r"^[A-F0-9]{8}$")


# -----------------------
# Prompt loading
# -----------------------
def load_prompts(prompt_dir: str, system_file: str, user_file: str) -> Tuple[str, str]:
    p = Path(prompt_dir)
    system_path = p / system_file
    user_path = p / user_file
    if not system_path.exists():
        raise FileNotFoundError(f"Missing system prompt: {system_path}")
    if not user_path.exists():
        raise FileNotFoundError(f"Missing user prompt: {user_path}")

    system = system_path.read_text(encoding="utf-8").strip()
    user = user_path.read_text(encoding="utf-8").strip()
    if "{filename}" not in user:
        raise ValueError("User prompt must contain {filename} placeholder.")
    return system, user


SYSTEM_PROMPT, USER_TEMPLATE = load_prompts(PROMPT_DIR, SYSTEM_FILE, USER_FILE)


def build_chat(system: str, user: str) -> str:
    return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"


# -----------------------
# Stop criteria
# -----------------------
class StopOnBalancedJSON(StoppingCriteria):
    """
    Stops when the GENERATED suffix contains a balanced {...} JSON object.
    Uses brace balancing to avoid waiting for json.loads to succeed mid-stream.
    Also stops if the model starts a new chat turn (<|system|>, <|user|>).
    """
    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0][self.prompt_len:]
        if gen_ids.numel() < 2:
            return False

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # If the model starts emitting new chat tags, stop immediately.
        if "<|system|>" in text or "<|user|>" in text:
            return True

        start = text.find("{")
        if start == -1:
            return False

        # Brace-balance scan from first '{'
        depth = 0
        for ch in text[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True  # first complete object closed

        return False


# -----------------------
# JSON extraction + validation
# -----------------------
def extract_first_balanced_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract the first balanced JSON object from text and parse it.
    """
    start = text.find("{")
    if start == -1:
        return None, "No '{' found."

    depth = 0
    end = None
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end is None:
        return None, "No matching '}' found (unbalanced braces)."

    candidate = text[start:end + 1].strip()
    try:
        return json.loads(candidate), ""
    except Exception as e:
        return None, f"JSON parse failed: {e}"


def validate_obj(obj: Dict[str, Any], require_episode: bool) -> List[str]:
    errors = []

    if list(obj.keys()) != EXPECTED_KEYS:
        errors.append(f"Keys/order mismatch. Expected {EXPECTED_KEYS}, got {list(obj.keys())}")

    if not isinstance(obj.get("show_name"), str) or not obj.get("show_name"):
        errors.append("show_name must be a non-empty string.")

    if obj.get("season") is not None and not isinstance(obj.get("season"), int):
        errors.append("season must be int or null.")

    if obj.get("episode") is not None and not isinstance(obj.get("episode"), int):
        errors.append("episode must be int or null.")

    if require_episode and obj.get("episode") is None:
        errors.append("episode is required but was null.")

    crc = obj.get("crc_hash")
    if crc is not None:
        if not isinstance(crc, str):
            errors.append("crc_hash must be string or null.")
        elif not CRC_RE.match(crc):
            errors.append("crc_hash must match ^[A-F0-9]{8}$ when present.")

    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)):
        errors.append("confidence must be a number.")
    else:
        c = float(conf)
        if c < 0.0 or c > 1.0:
            errors.append("confidence must be in [0, 1].")

    if not isinstance(obj.get("reasoning"), str):
        errors.append("reasoning must be a string.")

    return errors


# -----------------------
# Inference
# -----------------------
def main() -> None:
    print("[INFO] Starting infer_validate.py", flush=True)
    print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    print(f"[INFO] dtype={'bf16' if dtype == torch.bfloat16 else 'fp16'}", flush=True)

    print("[INFO] Loading base model...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=dtype,
        load_in_4bit=True,
        attn_implementation=ATTN_IMPL,
    )

    print("[INFO] Loading adapter...", flush=True)
    model.load_adapter(ADAPTER_DIR)
    FastLanguageModel.for_inference(model)

    print(f"[INFO] Loaded base: {BASE_MODEL}", flush=True)
    print(f"[INFO] Loaded adapter: {ADAPTER_DIR}", flush=True)
    print(f"[INFO] REQUIRE_EPISODE={REQUIRE_EPISODE} MAX_NEW_TOKENS={MAX_NEW_TOKENS}", flush=True)
    print(f"[INFO] #tests={len(TEST_FILENAMES)}\n", flush=True)

    for i, filename in enumerate(TEST_FILENAMES, 1):
        print("=" * 100, flush=True)
        print(f"[{i}] {filename}", flush=True)

        user = USER_TEMPLATE.format(filename=filename)
        prompt = build_chat(SYSTEM_PROMPT, user)

        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        stopping = StoppingCriteriaList([StopOnBalancedJSON(tokenizer, prompt_len)])

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                stopping_criteria=stopping,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        t1 = time.time()

        gen_ids = out[0][prompt_len:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"[INFO] gen_tokens={gen_ids.numel()} time={(t1 - t0):.2f}s", flush=True)
        if not decoded:
            print("[FAIL] Empty generation output.", flush=True)
            continue

        obj, err = extract_first_balanced_json(decoded)
        if obj is None:
            print("[FAIL] Could not extract JSON:", err, flush=True)
            print("Raw output:", flush=True)
            print(decoded, flush=True)
            continue

        errors = validate_obj(obj, require_episode=REQUIRE_EPISODE)
        if errors:
            print("[FAIL] Validation errors:", flush=True)
            for e in errors:
                print(" -", e, flush=True)
            print("Parsed object:", flush=True)
            print(json.dumps(obj, ensure_ascii=False, indent=2), flush=True)
            print("Raw output:", flush=True)
            print(decoded, flush=True)
        else:
            print("[PASS]", flush=True)
            print(json.dumps(obj, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL] Unhandled exception:", repr(e), flush=True)
        raise
