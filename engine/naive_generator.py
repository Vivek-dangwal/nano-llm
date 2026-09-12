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
model.eval()  # Set model to evaluation mode (turns off training-only features)

# 2. Prepare the input prompt
prompt = "Artificial Intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

max_new_tokens = 25
print(f"\nPrompt: '{prompt}'")
print(f"Generating {max_new_tokens} tokens naively (without caching)...")

start_time = time.time()

# 3. The raw generation loop
generated_tokens = input_ids
with torch.no_grad():  # Disables gradient calculation to save memory and CPU
    for step in range(max_new_tokens):
        # Pass the ENTIRE sequence through the model every single time
        outputs = model(generated_tokens)

        # Get logits (the model's raw unnormalized score for each word in vocabulary)
        # We only care about the scores at the very last position: [:, -1, :]
        next_token_logits = outputs.logits[:, -1, :]

        # Greedy decoding: pick the token ID with the highest score
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append newly predicted token to the sequence
        generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)

elapsed = time.time() - start_time
tokens_per_sec = max_new_tokens / elapsed

# 4. Decode the full output
output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print("\n--- Generation Result ---")
print(f"Output: {output_text}")
print(f"Time Taken: {elapsed:.2f} seconds")
print(f"Throughput: {tokens_per_sec:.2f} tokens/second")