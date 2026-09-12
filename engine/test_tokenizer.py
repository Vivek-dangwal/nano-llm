from transformers import AutoTokenizer

# Specify the model name
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

print(f"Downloading/loading tokenizer for: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Sample text to translate
sample_text = "Hello! I am building a custom LLM engine from scratch."

# Step A: Convert text into numerical token IDs
encoded_tokens = tokenizer.encode(sample_text)
print("\n--- Tokenization Result ---")
print(f"Original Text: {sample_text}")
print(f"Tokens (Numbers): {encoded_tokens}")
print(f"Total Tokens: {len(encoded_tokens)}")

# Step B: Convert numerical IDs back into words
decoded_text = tokenizer.decode(encoded_tokens)
print(f"Decoded Back: {decoded_text}")