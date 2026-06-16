#!/usr/bin/env python3
"""
实验 2: OpenAI 兼容 API Server 测试
=====================================

前置条件: 先启动 API Server
  docker compose up -d api-server

运行方式:
  pip install openai
  python scripts/02_api_server_test.py

学习目标:
  - 理解 vLLM API Server 的 OpenAI 兼容接口
  - 掌握 Chat Completion / Text Completion API
  - 学习流式输出、多轮对话、参数调节
"""

import time
from openai import OpenAI

# 连接到本地 vLLM API Server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # vLLM 默认无需 API key
)

MODEL = "facebook/opt-125m"

# ============================================================
# 1. 检查服务状态
# ============================================================
print("=" * 60)
print("1. 检查服务状态")
print("=" * 60)

# 模型列表
models = client.models.list()
for m in models.data:
    print(f"  可用模型: {m.id}")
print()

# ============================================================
# 2. Text Completion (文本补全)
# ============================================================
print("=" * 60)
print("2. Text Completion")
print("=" * 60)

completion = client.completions.create(
    model=MODEL,
    prompt="The capital of China is",
    max_tokens=30,
    temperature=0.0,
)
print(f"  Prompt: 'The capital of China is'")
print(f"  Output: {completion.choices[0].text!r}")
print(f"  Tokens: prompt={completion.usage.prompt_tokens}, "
      f"completion={completion.usage.completion_tokens}")
print()

# ============================================================
# 3. Chat Completion (聊天补全)
# ============================================================
print("=" * 60)
print("3. Chat Completion")
print("=" * 60)

# 注意: OPT-125M 不是 Instruct 模型，chat 效果有限
# 但可以验证 API 兼容性
chat_completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    max_tokens=30,
    temperature=0.0,
)
print(f"  Response: {chat_completion.choices[0].message.content!r}")
print()

# ============================================================
# 4. 流式输出 (Streaming)
# ============================================================
print("=" * 60)
print("4. 流式输出 (Streaming)")
print("=" * 60)

print("  流式输出: ", end="", flush=True)
stream = client.completions.create(
    model=MODEL,
    prompt="In the future, technology will",
    max_tokens=50,
    temperature=0.7,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end="", flush=True)
print("\n")

# ============================================================
# 5. 批量请求与并发
# ============================================================
print("=" * 60)
print("5. 批量请求")
print("=" * 60)

prompts = [
    "The color of the sky is",
    "Water boils at",
    "The Earth is",
    "2 + 2 equals",
]

start = time.time()
for prompt in prompts:
    completion = client.completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=20,
        temperature=0.0,
    )
    print(f"  Q: {prompt}")
    print(f"  A: {completion.choices[0].text!r}")
elapsed = time.time() - start
print(f"  总耗时: {elapsed:.2f}s")
print()

# ============================================================
# 6. 采样参数实验
# ============================================================
print("=" * 60)
print("6. API 采样参数实验")
print("=" * 60)

# temperature 对比
print("\n--- Temperature 对比 (via API) ---")
for temp in [0.0, 0.5, 1.0]:
    c = client.completions.create(
        model=MODEL,
        prompt="The meaning of life is",
        max_tokens=30,
        temperature=temp,
    )
    print(f"  temp={temp:.1f}: {c.choices[0].text!r}")

# top_p 对比
print("\n--- Top-P 对比 (via API) ---")
for top_p in [0.1, 0.5, 1.0]:
    c = client.completions.create(
        model=MODEL,
        prompt="The best thing about",
        max_tokens=30,
        temperature=0.7,
        top_p=top_p,
    )
    print(f"  top_p={top_p:.1f}: {c.choices[0].text!r}")

# frequency_penalty 对比
print("\n--- Frequency Penalty 对比 ---")
for fp in [0.0, 1.0, 2.0]:
    c = client.completions.create(
        model=MODEL,
        prompt="I like to eat",
        max_tokens=30,
        temperature=0.7,
        frequency_penalty=fp,
    )
    print(f"  freq_penalty={fp:.1f}: {c.choices[0].text!r}")

# ============================================================
# 7. 停止词测试
# ============================================================
print("\n" + "=" * 60)
print("7. 停止词测试")
print("=" * 60)

c = client.completions.create(
    model=MODEL,
    prompt="Count from 1 to 10:",
    max_tokens=100,
    temperature=0.0,
    stop=["5"],  # 遇到 "5" 时停止
)
print(f"  With stop=['5']: {c.choices[0].text!r}")
print(f"  Finish reason: {c.choices[0].finish_reason}")

print("\n实验完成!")
