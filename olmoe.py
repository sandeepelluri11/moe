import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "allenai/OLMoE-1B-7B-0924"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

prompt = "Mixture of experts models are"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8
    )

#print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(model.hf_device_map)
