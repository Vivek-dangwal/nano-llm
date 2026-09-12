import os
import sys
import time
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from engine.core import MiniLLMEngine

def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """Calculates total memory occupied by model parameters and buffers in MB."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    return total_bytes / (1024 * 1024)

def run_benchmark():
    print("=" * 60)
    print("        nano-llm INFERENCE & QUANTIZATION BENCHMARK         ")
    print("=" * 60)

    test_prompt = "Explain how the key-value cache speeds up transformer inference."
    test_messages = [{"role": "user", "content": test_prompt}]
    tokens_to_generate = 30

    # 1. FP32 Baseline
    print("\n[1/2] Benchmarking FP32 Model (quantize=False)...")
    engine_fp32 = MiniLLMEngine(quantize=False, debug=False)
    fp32_size_mb = calculate_model_size_mb(engine_fp32.model)

    # Warm-up
    for _ in engine_fp32.generate_chat_stream(test_messages, max_new_tokens=5, use_retrieval=False):
        pass

    # Timed run
    t0 = time.perf_counter()
    fp32_tokens = 0
    for _ in engine_fp32.generate_chat_stream(test_messages, max_new_tokens=tokens_to_generate, use_retrieval=False):
        fp32_tokens += 1
    fp32_time = time.perf_counter() - t0
    fp32_speed = fp32_tokens / max(fp32_time, 1e-4)

    print(f" -> Tensor Size:       {fp32_size_mb:.2f} MB")
    print(f" -> Generation Speed:  {fp32_speed:.2f} tok/s ({fp32_tokens} tokens in {fp32_time:.2f}s)")

    del engine_fp32

    # 2. Custom INT8 Quantized
    print("\n[2/2] Benchmarking Custom INT8 Model (quantize=True)...")
    engine_int8 = MiniLLMEngine(quantize=True, debug=False)
    int8_size_mb = calculate_model_size_mb(engine_int8.model)

    # Warm-up
    for _ in engine_int8.generate_chat_stream(test_messages, max_new_tokens=5, use_retrieval=False):
        pass

    # Timed run
    t0 = time.perf_counter()
    int8_tokens = 0
    for _ in engine_int8.generate_chat_stream(test_messages, max_new_tokens=tokens_to_generate, use_retrieval=False):
        int8_tokens += 1
    int8_time = time.perf_counter() - t0
    int8_speed = int8_tokens / max(int8_time, 1e-4)

    print(f" -> Tensor Size:       {int8_size_mb:.2f} MB")
    print(f" -> Generation Speed:  {int8_speed:.2f} tok/s ({int8_tokens} tokens in {int8_time:.2f}s)")

    # Report
    mem_reduction = ((fp32_size_mb - int8_size_mb) / fp32_size_mb) * 100

    print("\n" + "=" * 60)
    print("                     FINAL RESULTS REPORT                  ")
    print("=" * 60)
    print(f"{'Metric':<25} | {'FP32 Baseline':<15} | {'Custom INT8':<15}")
    print("-" * 60)
    print(f"{'Model Tensor Size':<25} | {fp32_size_mb:<12.1f} MB | {int8_size_mb:<12.1f} MB")
    print(f"{'Decoding Throughput':<25} | {fp32_speed:<12.2f} tok/s| {int8_speed:<12.2f} tok/s")
    print("-" * 60)
    print(f"Memory Reduction: {mem_reduction:.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()