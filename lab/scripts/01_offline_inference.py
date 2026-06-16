#!/usr/bin/env python3
"""
实验 1: vLLM 离线推理基础
===========================

运行方式:
  docker compose run offline-inference /app/scripts/01_offline_inference.py

学习目标:
  - 理解 LLM 类的基本用法
  - 掌握 SamplingParams 参数调节
  - 观察不同采样参数对输出的影响
"""

from vllm import LLM, SamplingParams

# ============================================================
# 1. 创建 LLM 实例 — 加载模型
# ============================================================
print("=" * 60)
print("1. 创建 LLM 实例")
print("=" * 60)

llm = LLM(
    model="facebook/opt-125m",    # OPT-125M: 仅 ~250MB, CPU 友好
    device="cpu",                  # CPU 模式
    dtype="float32",               # CPU 下必须 float32
    max_model_len=512,             # 限制最大序列长度（省内存）
    max_num_batched_tokens=512,    # 限制每批 token 数
    max_num_seqs=4,                # 限制并发请求数
    gpu_memory_utilization=0.9,    # CPU 模式下此参数不生效
    enforce_eager=True,            # CPU 模式不支持 CUDA Graph
)
print("模型加载完成!\n")

# ============================================================
# 2. 基础推理 — 单条 prompt
# ============================================================
print("=" * 60)
print("2. 基础推理 — 单条 prompt")
print("=" * 60)

prompt = "Once upon a time, there was a"
sampling_params = SamplingParams(
    temperature=0.0,    # 贪心解码，最确定的输出
    max_tokens=64,
)

outputs = llm.generate([prompt], sampling_params)
for output in outputs:
    print(f"  Prompt: {output.prompt!r}")
    print(f"  Output: {output.outputs[0].text!r}")
    print(f"  Tokens: {len(output.outputs[0].token_ids)}")
print()

# ============================================================
# 3. 批量推理 — 多条 prompt
# ============================================================
print("=" * 60)
print("3. 批量推理 — 多条 prompt")
print("=" * 60)

prompts = [
    "The capital of France is",
    "In the year 2050,",
    "The meaning of life is",
    "Python is a programming language that",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=64,
)

outputs = llm.generate(prompts, sampling_params)
for prompt, output in zip(prompts, outputs):
    print(f"  Q: {prompt}")
    print(f"  A: {output.outputs[0].text}")
    print()
print()

# ============================================================
# 4. 采样参数对比实验
# ============================================================
print("=" * 60)
print("4. 采样参数对比实验")
print("=" * 60)

test_prompt = "The future of artificial intelligence is"

# 4a. temperature 对比
print("\n--- Temperature 对比 ---")
for temp in [0.0, 0.3, 0.7, 1.0, 1.5]:
    sp = SamplingParams(temperature=temp, max_tokens=40, seed=42)
    out = llm.generate([test_prompt], sp)
    print(f"  temp={temp:.1f}: {out[0].outputs[0].text!r}")

# 4b. top_p 对比
print("\n--- Top-P 对比 ---")
for top_p in [0.1, 0.5, 0.9, 1.0]:
    sp = SamplingParams(temperature=0.7, top_p=top_p, max_tokens=40, seed=42)
    out = llm.generate([test_prompt], sp)
    print(f"  top_p={top_p:.1f}: {out[0].outputs[0].text!r}")

# 4c. top_k 对比
print("\n--- Top-K 对比 ---")
for top_k in [1, 5, 10, 50, -1]:
    sp = SamplingParams(temperature=0.7, top_k=top_k, max_tokens=40, seed=42)
    out = llm.generate([test_prompt], sp)
    label = f"top_k={top_k}" if top_k > 0 else "top_k=-1 (all)"
    print(f"  {label}: {out[0].outputs[0].text!r}")

# 4d. presence/frequency penalty 对比
print("\n--- 重复惩罚对比 ---")
for penalty in [0.0, 0.5, 1.0, 2.0]:
    sp = SamplingParams(
        temperature=0.7,
        presence_penalty=penalty,
        max_tokens=60,
        seed=42,
    )
    out = llm.generate([test_prompt], sp)
    print(f"  presence_penalty={penalty:.1f}: {out[0].outputs[0].text!r}")

# ============================================================
# 5. 停止词与 max_tokens
# ============================================================
print("\n" + "=" * 60)
print("5. 停止词与 max_tokens")
print("=" * 60)

prompt_story = "Tell me a story about a cat. Once upon a time"

# 使用 stop token
sp_stop = SamplingParams(
    temperature=0.7,
    max_tokens=200,
    stop=[".", ",", "\n"],  # 遇到句号、逗号或换行时停止
)
out = llm.generate([prompt_story], sp_stop)
print(f"  With stop=['.', ',', '\\n']: {out[0].outputs[0].text!r}")
print(f"  Finish reason: {out[0].outputs[0].finish_reason}")

# 使用 max_tokens 限制
sp_short = SamplingParams(temperature=0.7, max_tokens=10)
out = llm.generate([prompt_story], sp_short)
print(f"\n  With max_tokens=10: {out[0].outputs[0].text!r}")
print(f"  Finish reason: {out[0].outputs[0].finish_reason}")

# ============================================================
# 6. 多输出 (n > 1)
# ============================================================
print("\n" + "=" * 60)
print("6. 多输出 (n > 1)")
print("=" * 60)

sp_n = SamplingParams(temperature=0.8, max_tokens=40, n=3)
out = llm.generate(["The best programming language is"], sp_n)
for i, completion in enumerate(out[0].outputs):
    print(f"  Output {i}: {completion.text!r}")

print("\n实验完成!")
