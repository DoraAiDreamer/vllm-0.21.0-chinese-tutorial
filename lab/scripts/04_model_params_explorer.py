#!/usr/bin/env python3
"""
实验 4: vLLM 引擎参数深度探索
===============================

运行方式:
  docker compose run offline-inference /app/scripts/04_model_params_explorer.py

学习目标:
  - 理解 EngineArgs → VllmConfig 的参数传递链路
  - 观察 max_model_len 对内存的影响
  - 体验不同 block_size 的效果
  - 学习 KV Cache 配置
"""

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig

# ============================================================
# 1. 默认配置观察
# ============================================================
print("=" * 60)
print("1. 默认 VllmConfig 观察")
print("=" * 60)

llm = LLM(
    model="facebook/opt-125m",
    device="cpu",
    dtype="float32",
    max_model_len=256,
    enforce_eager=True,
)

# 获取 VllmConfig
config: VllmConfig = llm.llm_engine.model_executor.driver_worker.vllm_config

print(f"  model_config.model: {config.model_config.model}")
print(f"  model_config.dtype: {config.model_config.dtype}")
print(f"  model_config.max_model_len: {config.model_config.max_model_len}")
print(f"  cache_config.block_size: {config.cache_config.block_size}")
print(f"  cache_config.num_gpu_blocks: {config.cache_config.num_gpu_blocks}")
print(f"  cache_config.num_cpu_blocks: {config.cache_config.num_cpu_blocks}")
print(f"  scheduler_config.max_num_batched_tokens: "
      f"{config.scheduler_config.max_num_batched_tokens}")
print(f"  scheduler_config.max_num_seqs: {config.scheduler_config.max_num_seqs}")
print(f"  parallel_config.tensor_parallel_size: "
      f"{config.parallel_config.tensor_parallel_size}")
print()

# ============================================================
# 2. max_model_len 对比
# ============================================================
print("=" * 60)
print("2. max_model_len 对比")
print("=" * 60)

for max_len in [128, 256, 512]:
    try:
        test_llm = LLM(
            model="facebook/opt-125m",
            device="cpu",
            dtype="float32",
            max_model_len=max_len,
            enforce_eager=True,
        )
        cfg = test_llm.llm_engine.model_executor.driver_worker.vllm_config
        print(f"  max_model_len={max_len}: "
              f"num_gpu_blocks={cfg.cache_config.num_gpu_blocks}, "
              f"num_cpu_blocks={cfg.cache_config.num_cpu_blocks}")
        del test_llm
    except Exception as e:
        print(f"  max_model_len={max_len}: ERROR - {e}")
print()

# ============================================================
# 3. block_size 对比
# ============================================================
print("=" * 60)
print("3. block_size 对比")
print("=" * 60)

for block_size in [8, 16, 32]:
    try:
        test_llm = LLM(
            model="facebook/opt-125m",
            device="cpu",
            dtype="float32",
            max_model_len=256,
            block_size=block_size,
            enforce_eager=True,
        )
        cfg = test_llm.llm_engine.model_executor.driver_worker.vllm_config
        print(f"  block_size={block_size}: "
              f"num_blocks={cfg.cache_config.num_gpu_blocks}, "
              f"block_size={cfg.cache_config.block_size}")
        del test_llm
    except Exception as e:
        print(f"  block_size={block_size}: ERROR - {e}")
print()

# ============================================================
# 4. SamplingParams 与模型配置的交互
# ============================================================
print("=" * 60)
print("4. SamplingParams 与 max_tokens 限制")
print("=" * 60)

# max_tokens 超过 max_model_len 会被自动截断
llm_long = LLM(
    model="facebook/opt-125m",
    device="cpu",
    dtype="float32",
    max_model_len=64,  # 极小值
    enforce_eager=True,
)

# 尝试请求超过 max_model_len 的 max_tokens
try:
    sp = SamplingParams(max_tokens=200)  # 超过 max_model_len=64
    out = llm_long.generate(["Hello"], sp)
    actual_tokens = len(out[0].outputs[0].token_ids)
    print(f"  请求 max_tokens=200, max_model_len=64")
    print(f"  实际生成 token 数: {actual_tokens}")
    print(f"  Finish reason: {out[0].outputs[0].finish_reason}")
except Exception as e:
    print(f"  Error: {e}")

del llm_long
print()

# ============================================================
# 5. 多轮推理与 KV Cache 复用
# ============================================================
print("=" * 60)
print("5. 多轮推理实验")
print("=" * 60)

# 每次独立 generate 不共享 KV Cache
sp = SamplingParams(temperature=0.0, max_tokens=20)

# 第一轮
out1 = llm.generate(["The capital of France is"], sp)
print(f"  Round 1: {out1[0].outputs[0].text!r}")

# 第二轮（独立请求，不共享上下文）
out2 = llm.generate(["The capital of France is Paris. The capital of Germany is"], sp)
print(f"  Round 2: {out2[0].outputs[0].text!r}")
print()
print("  注意: vLLM 离线推理每次 generate 是独立请求，不共享 KV Cache")
print("  如需多轮对话，需在单次 generate 中传入完整历史 prompt")
print()

# ============================================================
# 6. 请求级别统计信息
# ============================================================
print("=" * 60)
print("6. 输出统计信息")
print("=" * 60)

sp = SamplingParams(temperature=0.7, max_tokens=30, logprobs=5)
out = llm.generate(["The future of AI is"], sp)

output = out[0].outputs[0]
print(f"  Text: {output.text!r}")
print(f"  Token count: {len(output.token_ids)}")
print(f"  Finish reason: {output.finish_reason}")
if output.logprobs:
    print(f"  Logprobs (first 5 tokens):")
    for i, lp in enumerate(output.logprobs[:5]):
        top_tokens = [(t.decoded_token, t.logprob) for t in lp.values()]
        print(f"    Token {i}: {top_tokens[:3]}")
print()

print("实验完成!")
