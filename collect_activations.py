#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------- CONFIG --------
MODEL_NAME = "Qwen/Qwen3-30B-A3B"
OUT_DIR = "/blue/ranka/sandeepelluri/moe_acts_all"
NUM_SEQUENCES = 64
BATCH_SIZE = 1
FLUSH_EVERY = 16
MODEL_DTYPE = torch.bfloat16
ACT_DTYPE = torch.float16
MAX_SEQ_LEN = 512  # truncate C4 sequences

# Offload config for large model
device_map = "auto"
max_memory = {
    0: "20GB",       # GPU 0
    "cpu": "240GB",  # rest goes to CPU
}
offload_folder = OUT_DIR + "/offload"

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(offload_folder).mkdir(parents=True, exist_ok=True)

# -------- LOAD MODEL & TOKENIZER --------
print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print("[INFO] Loading model with CPU offload...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    max_memory=max_memory,
    offload_folder=offload_folder,
    torch_dtype=MODEL_DTYPE,
)
model.eval()

# -------- LOAD DATASET --------
print("[INFO] Loading C4 dataset...")
dataset = load_dataset("allenai/c4", "en", split="train[:{}]".format(NUM_SEQUENCES))
sequences = []
for item in dataset:
    text = item["text"]
    tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
    sequences.append(tokens["input_ids"].squeeze(0))
print(f"[INFO] Loaded {len(sequences)} sequences.")

# -------- COLLECT ACTIVATIONS --------
def save_activations(layer_idx, activations):
    path = Path(OUT_DIR) / f"layer_{layer_idx}_router_input.npy"
    np.save(path, activations)
    print(f"[INFO] Saved activations: {path}")

print("[INFO] Starting activation collection...")

with torch.no_grad():
    for seq_idx, input_ids in enumerate(sequences):
        input_ids = input_ids.unsqueeze(0)  # batch dimension
        input_ids = input_ids.to(model.device)

        # Forward pass and collect hidden states
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of layer activations

        # Convert to CPU + desired dtype
        for layer_idx, layer_act in enumerate(hidden_states):
            layer_act = layer_act.cpu().to(ACT_DTYPE).numpy()
            save_activations(layer_idx, layer_act)

        if (seq_idx + 1) % FLUSH_EVERY == 0:
            print(f"[INFO] Flushed activations for {seq_idx + 1} sequences.")

print("[INFO] Done collecting activations!")
