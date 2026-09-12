import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

print(f"Downloading and loading model weights for: {MODEL_ID}...")
# We load the model in torch.float32 for standard CPU execution
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# Calculate total number of parameters (the weights)
total_params = sum(p.numel() for p in model.parameters())
print("\n--- Model Inspection ---")
print(f"Model Architecture: {model.__class__.__name__}")
print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.1f} Million)")
print(f"Hidden Dimension Size: {model.config.hidden_size}")
print(f"Number of Transformer Layers: {model.config.num_hidden_layers}")
print(f"Number of Attention Heads: {model.config.num_attention_heads}")