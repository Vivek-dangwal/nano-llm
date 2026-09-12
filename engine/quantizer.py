import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

class QuantizedLinear(nn.Module):
    """
    A custom linear layer that stores weights as 8-bit signed integers (INT8)
    along with a per-tensor scaling factor, dequantizing dynamically during forward pass.
    """
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # 1. Compute scaling factor: max absolute value / 127
        w = original_linear.weight.data
        max_val = torch.max(torch.abs(w))
        self.scale = max_val / 127.0

        # 2. Quantize FP32 -> INT8 (-128 to 127)
        q_weight = torch.clamp(torch.round(w / self.scale), -128, 127).to(torch.int8)
        self.register_buffer("q_weight", q_weight)

        # 3. Preserve bias if present
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize back to FP32 for the matrix multiplication
        dequant_weight = self.q_weight.to(x.dtype) * self.scale
        return nn.functional.linear(x, dequant_weight, self.bias)


def quantize_model_layers(model: nn.Module):
    """
    Recursively traverses the transformer model and swaps standard nn.Linear
    projections (Q, K, V, Out, MLP) with our custom QuantizedLinear layers.
    """
    for name, child in model.named_children():
        # We quantize the attention and feed-forward linear layers, keeping lm_head intact
        if isinstance(child, nn.Linear) and name != "lm_head":
            setattr(model, name, QuantizedLinear(child))
        else:
            quantize_model_layers(child)


if __name__ == "__main__":
    print("Loading standard FP32 model...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # Calculate weight memory of original model
    fp32_mem_mb = sum(p.numel() * p.element_size() for p in model_fp32.parameters()) / (1024 * 1024)
    print(f"Original Model Weight Memory: {fp32_mem_mb:.2f} MB")

    print("\nApplying custom INT8 quantization to linear layers...")
    quantize_model_layers(model_fp32.model)  # Target the transformer body

    # Calculate new memory (parameters + INT8 buffers)
    param_mem = sum(p.numel() * p.element_size() for p in model_fp32.parameters())
    buffer_mem = sum(b.numel() * b.element_size() for b in model_fp32.buffers())
    quant_mem_mb = (param_mem + buffer_mem) / (1024 * 1024)

    print(f"Quantized Model Memory: {quant_mem_mb:.2f} MB")
    print(f"Memory Reduction: {((fp32_mem_mb - quant_mem_mb) / fp32_mem_mb) * 100:.1f}% savings")

    # Verification inference
    print("\nVerifying generation quality on quantized weights...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt = "The three primary states of matter are"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model_fp32.generate(**inputs, max_new_tokens=20)

    print("Decoded Output:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))