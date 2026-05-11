"""Quick smoke test for the training environment.

Verifies that PyTorch sees a CUDA device, that a representative tokenizer
loads, that the Hugging Face Datasets cache is reachable, and that PEFT is
importable. Run this once on any new node before launching real training.
"""

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print("Tokenizer loaded.")

ds = load_dataset("openai/gsm8k", "main", split="train[:5]")
print("Dataset loaded:", len(ds))

cfg = LoraConfig(r=8, lora_alpha=16)
print("PEFT config ok.")