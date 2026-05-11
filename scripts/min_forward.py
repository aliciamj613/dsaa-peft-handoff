"""Minimal end-to-end forward pass on the base model.

Used as a sanity check to confirm that the model weights load on GPU and that
greedy generation works before launching anything heavier (training, eval).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-3B-Instruct"

tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

text = "What is 2 + 2?"
inputs = tok(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)

print(tok.decode(out[0], skip_special_tokens=True))