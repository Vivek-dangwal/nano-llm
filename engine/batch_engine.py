import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# SmolLM2 tokenizer requires an explicit pad token for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    low_cpu_mem_usage=True
)
model.eval()

# Three different incoming requests arriving simultaneously
prompts = [
    "Python is a programming language that",
    "The capital of France is",
    "Explain gravity in simple terms:"
]

print(f"\nIncoming Batch Requests ({len(prompts)}):")
for i, p in enumerate(prompts):
    print(f"  Req {i+1}: '{p}'")

# Tokenize all prompts together with left-padding
# In causal text generation, padding must be on the LEFT so the actual sequence ends together
tokenizer.padding_side = "left"
encoded_inputs = tokenizer(prompts, return_tensors="pt", padding=True)

input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]

batch_size = input_ids.shape[0]
max_new_tokens = 20

print(f"\nExecuting Batched Generation for {batch_size} sequences...")
start_time = time.time()

generated_ids = input_ids
with torch.no_grad():
    # Prefill phase with attention mask to ignore pad tokens
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values
    next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

    # Decode phase: compute next token for all 3 sequences in one forward pass
    for _ in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_tokens,
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        next_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

total_time = time.time() - start_time
total_tokens_generated = batch_size * max_new_tokens
system_throughput = total_tokens_generated / total_time

print("\n--- Batched Inference Results ---")
for i, gen_seq in enumerate(generated_ids):
    result_text = tokenizer.decode(gen_seq, skip_special_tokens=True)
    print(f"\nResult [{i+1}]:\n{result_text}")

print(f"\nTotal Time: {total_time:.2f} seconds")
print(f"Aggregated System Throughput: {system_throughput:.2f} tokens/second")