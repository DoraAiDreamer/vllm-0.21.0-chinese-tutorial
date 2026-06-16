#!/usr/bin/env python3
"""
实验 3: vLLM 日志系统探索
===========================

运行方式:
  docker compose run -e VLLM_LOGGING_LEVEL=DEBUG offline-inference /app/scripts/03_logging_experiments.py

学习目标:
  - 理解 VLLM_LOGGING_LEVEL 对日志的影响
  - 观察不同日志级别下的输出差异
  - 学习自定义日志配置文件用法
  - 掌握统计日志间隔控制
"""

import os
import logging
import logging.config

# ============================================================
# 1. 当前日志环境变量检查
# ============================================================
print("=" * 60)
print("1. 当前日志环境变量")
print("=" * 60)

log_vars = [
    "VLLM_LOGGING_LEVEL",
    "VLLM_LOGGING_PREFIX",
    "VLLM_LOGGING_STREAM",
    "VLLM_LOGGING_COLOR",
    "VLLM_CONFIGURE_LOGGING",
    "VLLM_LOGGING_CONFIG_PATH",
    "VLLM_LOG_STATS_INTERVAL",
    "VLLM_LOG_BATCHSIZE_INTERVAL",
    "VLLM_LOG_MODEL_INSPECTION",
]

for var in log_vars:
    val = os.environ.get(var, "(not set)")
    print(f"  {var} = {val}")
print()

# ============================================================
# 2. Python logging 配置检查
# ============================================================
print("=" * 60)
print("2. Python logging 配置")
print("=" * 60)

vllm_logger = logging.getLogger("vllm")
print(f"  vllm logger level: {vllm_logger.level} "
      f"({logging.getLevelName(vllm_logger.level)})")
print(f"  vllm logger handlers: {vllm_logger.handlers}")
if vllm_logger.handlers:
    for h in vllm_logger.handlers:
        print(f"    - {h.__class__.__name__}: level={h.level}, "
              f"formatter={h.formatter.__class__.__name__}")
print()

# ============================================================
# 3. 使用 LLM 观察日志输出
# ============================================================
print("=" * 60)
print("3. LLM 加载过程日志观察")
print("=" * 60)
print("(以下日志来自 vllm 内部 logger，观察 INFO 级别输出)\n")

from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-125m",
    device="cpu",
    dtype="float32",
    max_model_len=256,
    max_num_batched_tokens=256,
    max_num_seqs=2,
    enforce_eager=True,
)
print("\n模型加载完成\n")

# ============================================================
# 4. 推理过程中的日志
# ============================================================
print("=" * 60)
print("4. 推理过程日志观察")
print("=" * 60)
print("(观察每次 generate 调用时 vllm 内部日志输出)\n")

prompts = [
    "Hello, world!",
    "The meaning of life is",
    "In the year 2050,",
]

sp = SamplingParams(temperature=0.7, max_tokens=30)
outputs = llm.generate(prompts, sp)

for prompt, output in zip(prompts, outputs):
    print(f"  Q: {prompt}")
    print(f"  A: {output.outputs[0].text!r}")
print()

# ============================================================
# 5. 自定义日志前缀实验
# ============================================================
print("=" * 60)
print("5. 日志前缀实验")
print("=" * 60)

# 在进程内修改前缀（仅演示，实际需在启动前设置）
import vllm.envs as envs
old_prefix = envs.VLLM_LOGGING_PREFIX
print(f"  当前日志前缀: {old_prefix!r}")
print("  提示: 可通过 VLLM_LOGGING_PREFIX 环境变量设置自定义前缀")
print("  例如: VLLM_LOGGING_PREFIX='[my-lab] '")
print()

# ============================================================
# 6. 统计日志间隔实验
# ============================================================
print("=" * 60)
print("6. 统计日志间隔控制")
print("=" * 60)
print("  VLLM_LOG_STATS_INTERVAL 环境变量控制统计日志间隔（秒）")
print("  默认值: 10.0")
print("  设为 -1 禁用定期统计")
print("  设为更小值（如 5）获得更频繁的统计")
print()

# ============================================================
# 7. 日志配置文件说明
# ============================================================
print("=" * 60)
print("7. 自定义日志配置文件")
print("=" * 60)
print("  路径: /app/configs/vllm_logging.json")
print("  使用方法:")
print("    export VLLM_LOGGING_CONFIG_PATH=/app/configs/vllm_logging.json")
print("    export VLLM_CONFIGURE_LOGGING=True")
print()
print("  配置文件可控制:")
print("    - 日志格式 (formatters)")
print("    - 输出位置 (handlers: console / file / rotating file)")
print("    - 日志级别 (level)")
print("    - 日志颜色 (ColoredFormatter)")
print()

# ============================================================
# 8. DEBUG 级别下的差异
# ============================================================
print("=" * 60)
print("8. DEBUG 级别的差异")
print("=" * 60)
print("  当 VLLM_LOGGING_LEVEL=DEBUG 时:")
print("    - 文件路径显示为缩写形式（而非仅文件名）")
print("    - 输出更多内部调试信息")
print("    - 如: model_executor/.../quantization/utils/fp8_utils.py")
print("  当 VLLM_LOGGING_LEVEL=INFO 时:")
print("    - 文件路径仅显示文件名")
print("    - 如: fp8_utils.py")
print()

# ============================================================
# 9. 一次性日志 (info_once / warning_once)
# ============================================================
print("=" * 60)
print("9. 一次性日志机制")
print("=" * 60)
print("  vLLM 提供 debug_once / info_once / warning_once 方法")
print("  相同消息只输出一次，避免日志刷屏")
print()
print("  支持分布式作用域:")
print("    scope='process' — 所有进程都输出（默认）")
print("    scope='local'   — 仅本地首个 rank 输出")
print("    scope='global'  — 仅全局首个 rank 输出")
print()

# ============================================================
# 10. 函数调用追踪（调试 hang/crash）
# ============================================================
print("=" * 60)
print("10. 函数调用追踪")
print("=" * 60)
print("  vLLM 提供函数调用追踪功能，用于调试 hang 或崩溃:")
print()
print("  from vllm.logger import enable_trace_function_call")
print("  enable_trace_function_call('/path/to/trace.log')")
print()
print("  注意: 此功能会大幅降低性能，仅用于调试!")
print()

print("实验完成!")
