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


def configure_logging(name: str = "infer_validate", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a structured logger with optional file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


logger = configure_logging()


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
# Stop on chat tags to avoid runaway chat re-entries.
STOP_ON_CHAT_TAGS = True

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


def build_inputs(tokenizer, system: str, user: str) -> Dict[str, torch.Tensor]:
    """
    Build model inputs using the tokenizer chat template when available.
    Falls back to the manual format used during training if the template is missing.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ""},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        pad_id = tokenizer.pad_token_id
        attn_mask = torch.ones_like(tokenized, dtype=torch.long) if pad_id is None else tokenized.ne(pad_id).long()
        return {"input_ids": tokenized, "attention_mask": attn_mask}

    # Fallback to manual string format (matches training).
    prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    return tokenizer(prompt, return_tensors="pt")


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

        # Optionally stop early on chat tags (disabled by default).
        if STOP_ON_CHAT_TAGS and any(tag in text for tag in ("<|system|>", "<|user|>", "<|assistant|>")):
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


def generate_once(
    model,
    tokenizer,
    inputs: Dict[str, torch.Tensor],
    stopping: StoppingCriteriaList,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[str, int]:
    """Single generation attempt."""
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.0,
            stopping_criteria=stopping,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return decoded, gen_ids.numel()


def generate_with_retry(
    model,
    tokenizer,
    inputs: Dict[str, torch.Tensor],
    stopping: StoppingCriteriaList,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Generate with retries and basic validation."""
    attempts = [
        (0.0, 1.0),   # greedy
        (0.1, 0.95),  # mild sampling
        (0.2, 0.9),   # slightly more sampling
        (0.3, 0.85),  # fallback: more exploratory
    ][:max_retries]

    if not attempts:
        return {
            "status": "fail",
            "attempt": 0,
            "temperature": None,
            "top_p": None,
            "gen_tokens": 0,
            "decoded": "",
            "obj": None,
            "errors": ["no generation attempts: max_retries <= 0"],
        }

    best = None
    for idx, (temp, top_p) in enumerate(attempts, 1):
        decoded, gen_tokens = generate_once(model, tokenizer, inputs, stopping, temp, top_p, max_new_tokens)
        obj, err = extract_first_balanced_json(decoded)
        val_errors = validate_obj(obj, require_episode=REQUIRE_EPISODE) if obj else [err]
        if obj and not val_errors:
            return {
                "status": "ok",
                "attempt": idx,
                "temperature": temp,
                "top_p": top_p,
                "gen_tokens": gen_tokens,
                "decoded": decoded,
                "obj": obj,
                "errors": [],
            }
        best = {
            "status": "fail",
            "attempt": idx,
            "temperature": temp,
            "top_p": top_p,
            "gen_tokens": gen_tokens,
            "decoded": decoded,
            "obj": obj,
            "errors": val_errors,
        }
    return best


# -----------------------
# Inference
# -----------------------
def run(benchmark: bool = False) -> None:
    logger.info("Starting infer_validate.py")
    logger.info("torch=%s cuda_available=%s", torch.__version__, torch.cuda.is_available())

    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    logger.info("dtype=%s", "bf16" if dtype == torch.bfloat16 else "fp16")

    logger.info("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=dtype,
        load_in_4bit=True,
        attn_implementation=ATTN_IMPL,
    )

    logger.info("Loading adapter...")
    model.load_adapter(ADAPTER_DIR)
    FastLanguageModel.for_inference(model)

    logger.info("Loaded base: %s", BASE_MODEL)
    logger.info("Loaded adapter: %s", ADAPTER_DIR)
    logger.info("REQUIRE_EPISODE=%s MAX_NEW_TOKENS=%s", REQUIRE_EPISODE, MAX_NEW_TOKENS)
    test_filenames = TEST_FILENAMES[:1] if benchmark else TEST_FILENAMES
    logger.info("#tests=%s", len(test_filenames))

    for i, filename in enumerate(test_filenames, 1):
        logger.info("=" * 100)
        logger.info("[%s] %s", i, filename)

        user = USER_TEMPLATE.format(filename=filename)
        inputs = build_inputs(tokenizer, SYSTEM_PROMPT, user)
        prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        stopping = StoppingCriteriaList([StopOnBalancedJSON(tokenizer, prompt_len)])

        t0 = time.time()
        result = generate_with_retry(
            model,
            tokenizer,
            inputs,
            stopping,
            max_new_tokens=MAX_NEW_TOKENS,
            max_retries=3,
        )
        t1 = time.time()

        logger.info(
            "gen_tokens=%s time=%.2fs attempts=%s",
            result.get("gen_tokens", 0),
            (t1 - t0),
            result.get("attempt"),
        )

        if result["status"] != "ok":
            logger.error("[FAIL]")
            for e in result.get("errors", []):
                logger.error(" - %s", e)
            if result.get("obj"):
                logger.error("Parsed object (invalid): %s", json.dumps(result["obj"], ensure_ascii=False))
            logger.error("Raw output: %s", result.get("decoded", ""))
            continue

        logger.info("[PASS]")
        logger.info(json.dumps(result["obj"], ensure_ascii=False))


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run a short benchmark (single test filename).")
    args = parser.parse_args(argv)
    try:
        run(benchmark=args.benchmark)
    except Exception as e:
        logger.error("[FATAL] Unhandled exception: %s", repr(e))
        raise


if __name__ == "__main__":
    main()
