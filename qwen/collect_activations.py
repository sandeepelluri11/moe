#!/usr/bin/env python3

import os
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= CONFIG =================

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
OUT_DIR = "./router_activations"

NUM_SEQUENCES = 64
MAX_SEQ_LEN = 512
FLUSH_EVERY = 8

MODEL_DTYPE = torch.bfloat16
ACT_DTYPE = torch.float16

# B200 / H100 ? "cuda"
# L4 / A100 ? "auto"
DEVICE_MAP = "cuda"

os.environ["HF_DATASETS_CACHE"] = "./hf_cache"

# ==========================================

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ================= LOAD MODEL =================

print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print("[INFO] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=DEVICE_MAP,
    dtype=MODEL_DTYPE,
)

model.eval()

# ================= STREAMING DATASET =================

print("[INFO] Streaming C4 dataset...")

dataset = load_dataset(
    "allenai/c4",
    "en",
    split="train",
    streaming=True
)

# ================= ROUTER HOOK =================

router_buffers = defaultdict(list)

def create_router_hook(layer_name):

    def hook(module, inputs, outputs):
        try:
            x = inputs[0].detach().cpu().to(ACT_DTYPE)

            # ? Flatten batch + seq ? tokens
            x = x.reshape(-1, x.shape[-1]).numpy()

            router_buffers[layer_name].append(x)

        except Exception as e:
            print(f"[WARNING] Hook failed for {layer_name}: {e}")

    return hook

print("[INFO] Searching for MoE router layers...")

router_found = 0

for name, module in model.named_modules():

    # Qwen MoE detection via gate projection
    if hasattr(module, "gate"):

        module.register_forward_hook(create_router_hook(name))
        print(f"[HOOK] Router detected at: {name}")
        router_found += 1

if router_found == 0:
    raise RuntimeError("No MoE router modules detected!")

print(f"[INFO] Total router layers found: {router_found}")

# ================= SAVE FUNCTION =================

def flush_buffers():

    print("[INFO] Flushing activations to disk...")

    for layer_name, acts in router_buffers.items():

        if len(acts) == 0:
            continue

        try:
            stacked = np.concatenate(acts, axis=0)

            safe_name = layer_name.replace(".", "_")
            path = Path(OUT_DIR) / f"{safe_name}.npy"

            if path.exists():
                old = np.load(path)
                stacked = np.concatenate([old, stacked], axis=0)

            np.save(path, stacked)
            router_buffers[layer_name] = []

        except Exception as e:
            print(f"[WARNING] Flush failed for {layer_name}: {e}")

# ================= FORWARD LOOP =================

print("[INFO] Starting activation collection...")

with torch.no_grad():

    seq_counter = 0

    for item in dataset:

        if seq_counter >= NUM_SEQUENCES:
            break

        try:
            tokens = tokenizer(
                item["text"],
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt"
            )

            input_ids = tokens["input_ids"].to(model.device)

            _ = model(input_ids)

            seq_counter += 1

            if seq_counter % FLUSH_EVERY == 0:
                flush_buffers()

            print(f"[INFO] Processed {seq_counter}/{NUM_SEQUENCES}")

        except Exception as e:
            print(f"[WARNING] Skipping sample: {e}")

# Final flush
flush_buffers()

print("[INFO] Activation collection complete!")
