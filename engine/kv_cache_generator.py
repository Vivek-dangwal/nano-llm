import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# 1. Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
    low_cpu_mem_usage=True
)
model.eval()

prompt = "Artificial Intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
max_new_tokens = 25

print(f"\nPrompt: '{prompt}'")
print(f"Generating {max_new_tokens} tokens WITH KV-Caching...")

start_time = time.time()
generated_tokens = input_ids

with torch.no_grad():
    # --- PREFILL PHASE ---
    # Process the prompt once to get initial logits and build the initial KV-Cache
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

    # Get first predicted token
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)

    # --- DECODE PHASE ---
    # For every subsequent token, ONLY feed the single new token!
    # The model reuses past_key_values instead of recomputing past tokens.
    for step in range(max_new_tokens - 1):
        outputs = model(
            input_ids=next_token_id,  # Shape is [1, 1] — just ONE token!
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)

elapsed = time.time() - start_time
tokens_per_sec = max_new_tokens / elapsed

output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print("\n--- KV-Cache Generation Result ---")
print(f"Output: {output_text}")
print(f"Time Taken: {elapsed:.2f} seconds")
print(f"Throughput: {tokens_per_sec:.2f} tokens/second")