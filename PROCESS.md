# Process to create the model

## Run the SFT training script

This take roughly 40m on an RTX 4090.

```python
python sft_qwen3_metadata.py
```

## Validate the difficult synthetic cases
This will test the newly fine tuned model with some difficult edge cases.

```python
python infer_validate.py
```

## Add the model to Ollama for local inference

### Create the HF model obj

```shell
python merge_lora_to_full_model.py
```

### Download the source and binaries for llama.cpp

https://github.com/ggml-org/llama.cpp/releases

### Create a GGUF from the SFT
```shell
python D:\Languages\llama\llama.cpp-b7415\convert_hf_to_gguf.py D:\Documents\Code\LLM_Fine_Tuning\outputs\merged_hf_model --outfile D:\Documents\Code\LLM_Fine_Tuning\outputs\qwen3-animemetadata-f16.gguf --outtype f16
```

### Quantize the model
```shell
D:\Languages\llama\llama-b7415-bin-win-cuda-12.4-x64\llama-quantize.exe D:\Documents\Code\LLM_Fine_Tuning\outputs\qwen3-animemetadata-f16.gguf D:\Documents\Code\LLM_Fine_Tuning\outputs\qwen3-animemetadata-q4_k_m.gguf Q4_K_M
```

### Create the Modelfile pointing to the GGUF

### Create the ollama entry

```shell
ollama create qwen3-animemetadata -f Modelfile
```

### Use the model in ollama
