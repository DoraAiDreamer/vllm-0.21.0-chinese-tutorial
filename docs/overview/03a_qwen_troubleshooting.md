# Qwen 系列模型部署常见问题与排错

> 基于 vllm-0.21.0 版本

本文档整理 Qwen 系列模型在 vLLM 上部署时遇到的常见问题、踩坑记录及解决方案，覆盖 NVIDIA GPU 和华为昇腾 910B 两种硬件平台。

---

## 一、Qwen 模型家族概览

vLLM 0.21.0 支持的 Qwen 系列模型架构：

| 模型 | 架构名 | 类型 | 文件 |
|------|--------|------|------|
| Qwen (1.0) | `QWenLMHeadModel` | Decoder-only | `qwen.py` |
| Qwen1.5/2 | `Qwen2ForCausalLM` | Decoder-only | `qwen2.py` |
| Qwen2-MoE | `Qwen2MoeForCausalLM` | MoE | `qwen2_moe.py` |
| Qwen2-VL | `Qwen2VLForConditionalGeneration` | 视觉多模态 | `qwen2_vl.py` |
| Qwen2.5-VL | `Qwen2_5_VLForConditionalGeneration` | 视觉多模态 | `qwen2_5_vl.py` |
| Qwen2-Audio | `Qwen2AudioForConditionalGeneration` | 音频多模态 | `qwen2_audio.py` |
| Qwen2.5-Omni | `Qwen2_5OmniModel` | 全模态 | `qwen2_5_omni_thinker.py` |
| Qwen2-RM | `Qwen2ForRewardModel` | Reward | `qwen2_rm.py` |
| Qwen3 | `Qwen3ForCausalLM` | Decoder-only | `qwen3.py` |
| Qwen3-MoE | `Qwen3MoeForCausalLM` | MoE | `qwen3_moe.py` |
| Qwen3-VL | `Qwen3VLForConditionalGeneration` | 视觉多模态 | `qwen3_vl.py` |
| Qwen3-VL-MoE | `Qwen3VLMoeForConditionalGeneration` | MoE+视觉 | `qwen3_vl_moe.py` |
| Qwen3-ASR | `Qwen3ASRForConditionalGeneration` | 语音 | `qwen3_asr.py` |
| Qwen3.5 | `Qwen3_5ForConditionalGeneration` | 混合注意力+Mamba | `qwen3_5.py` |
| Qwen3.5-MoE | `Qwen3_5MoeForConditionalGeneration` | MoE+混合 | `qwen3_5.py` |
| Qwen3-Omni-MoE | `Qwen3OmniMoeForConditionalGeneration` | 全模态MoE | `qwen3_omni_moe_thinker.py` |
| Qwen3-Next | `Qwen3NextForCausalLM` | 下一代 | `qwen3_next.py` |
| Qwen-VL (1.0) | `QwenVLForConditionalGeneration` | 视觉多模态 | `qwen_vl.py` |

---

## 二、通用问题

### 2.1 `trust_remote_code` 未开启

**现象：**
```
ValueError: The repository for Qwen/Qwen-7B contains custom code which requires 
to execute arbitrary code. Please pass trust_remote_code=True.
```

**原因：** Qwen 1.0 系列在 HuggingFace 上使用自定义代码，需要显式信任。

**解决：**
```python
llm = LLM(model="Qwen/Qwen-7B", trust_remote_code=True)
```
```bash
vllm serve Qwen/Qwen-7B --trust-remote-code
```

> **注意：** Qwen2 及以后版本一般不需要此参数，但某些社区微调版本仍可能需要。

### 2.2 Tokenizer 不兼容 / 编码错误

**现象：**
```
ValueError: The model's tokenizer class is not supported
```
或生成输出包含乱码、重复 token。

**原因：**
- Qwen 1.0 使用 tiktoken 分词器，部分版本与 `tokenizer_mode="fast"` 不兼容
- Qwen2+ 使用标准 HuggingFace tokenizer，一般无此问题

**解决：**
```python
# Qwen 1.0 使用慢速 tokenizer
llm = LLM(model="Qwen/Qwen-7B", tokenizer_mode="slow", trust_remote_code=True)

# Qwen2+ 使用 auto（默认即可）
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", tokenizer_mode="auto")
```

### 2.3 Chat Template 缺失或不正确

**现象：** 生成的回复没有遵循对话格式，出现 `<|im_start|>` 等标记泄漏。

**原因：** 部分早期 Qwen 模型的 `tokenizer_config.json` 缺少 `chat_template` 字段，或自定义微调版本覆盖了模板。

**解决：**
```python
# 方法 1：指定 chat_template
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    chat_template="{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
)

# 方法 2：使用 API Server 时通过请求传入
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "chat_template": "..."
  }'
```

### 2.4 Stop Token 未正确设置

**现象：** 生成不停，直到达到 `max_tokens` 限制。

**原因：** Qwen2/2.5 使用 `<|im_end|>` 作为对话结束标记，而非常见的 `<|eot_id|>`。

**解决：**
```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    max_tokens=512,
    stop=["<|im_end|>", "<|endoftext|>"],
    stop_token_ids=[151645, 151643],  # Qwen2 的 stop token IDs
)
```

### 2.5 `max_model_len` 自动缩减

**现象：**
```
WARNING: The model's max seq len (32768) is larger than the maximum number of 
tokens that can fit in GPU memory. Reducing max_model_len to 16384.
```

**原因：** GPU 显存不足以支撑模型声称的最大长度对应的全量 KV Cache。

**解决：**
```python
# 方法 1：手动限制（推荐，可避免自动缩减的不确定性）
llm = LLM(model="Qwen/Qwen2.5-72B-Instruct", max_model_len=8192)

# 方法 2：降低 gpu_memory_utilization 腾出空间给 KV Cache（不推荐，可能 OOM）
llm = LLM(model="Qwen/Qwen2.5-72B-Instruct", gpu_memory_utilization=0.95)

# 方法 3：开启前缀缓存（对重复 prompt 场景有效）
llm = LLM(model="Qwen/Qwen2.5-72B-Instruct", enable_prefix_caching=True)

# 方法 4：使用 KV Cache 量化
llm = LLM(model="Qwen/Qwen2.5-72B-Instruct", kv_cache_dtype="fp8")
```

---

## 三、Qwen MoE 模型问题

### 3.1 Qwen2-MoE / Qwen3-MoE 显存不足

**现象：** MoE 模型参数量看似不大（如 Qwen2-57B-A14B），但实际显存占用远超同参数量 Dense 模型。

**原因：** MoE 模型虽有稀疏激活（仅激活部分专家），但所有专家权重需常驻显存。`57B-A14B` 表示总参数 57B，每次激活 14B，但加载时需 57B 全量权重。

**解决：**
```python
# 方法 1：使用 FP8 量化
llm = LLM(model="Qwen/Qwen2-57B-A14B", quantization="fp8")

# 方法 2：张量并行（至少 2 卡）
llm = LLM(model="Qwen/Qwen2-57B-A14B", tensor_parallel_size=2)

# 方法 3：专家并行（大规模部署推荐）
llm = LLM(
    model="Qwen/Qwen3-235B-A22B",
    tensor_parallel_size=8,
    enable_expert_parallel=True,  # 启用专家并行
)

# 方法 4：CPU 卸载（小显存卡）
llm = LLM(
    model="Qwen/Qwen2-57B-A14B",
    cpu_offload_gb=20,  # 将 20GB 权重卸载到 CPU
)
```

### 3.2 Qwen3-MoE 专家并行配置

**现象：** 大规模 MoE 模型（如 Qwen3-235B）在多卡上推理效率低。

**原因：** 默认仅使用张量并行，所有专家权重在每张卡上复制。

**解决：**
```bash
# 启用专家并行 — 专家权重分片到不同 GPU
vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

```python
llm = LLM(
    model="Qwen/Qwen3-235B-A22B",
    tensor_parallel_size=8,
    enable_expert_parallel=True,
)
```

**注意事项：**
- `enable_expert_parallel=True` 时，每个专家的权重只存放在一个 rank 上
- 需要确保 TP size 能被专家数整除，否则部分 rank 负载不均
- 专家并行与流水线并行不兼容

### 3.3 MoE Kernel 选择

**现象：** MoE 推理速度慢，或出现 kernel 报错。

**原因：** 默认 `moe_backend="auto"` 可能选择了次优后端。

**解决：**
```python
llm = LLM(
    model="Qwen/Qwen3-235B-A22B",
    moe_backend="triton",          # Triton 后端（通用性好）
    # moe_backend="deep_gemm",    # FP8 专用，H100/A100 推荐
    # moe_backend="cutlass",      # CUTLASS 后端
)
```

**MoE Backend 选项：**
| 后端 | 适用场景 | 限制 |
|------|----------|------|
| `auto` | 默认，自动选择 | — |
| `triton` | 通用，兼容性最好 | 性能非最优 |
| `deep_gemm` | FP8 量化模型，H100/H200 | 需要 FP8 量化 |
| `cutlass` | CUTLASS 优化 | 部分 GPU 不支持 |
| `flashinfer_cutlass` | FlashInfer + CUTLASS | 需安装 FlashInfer |
| `marlin` | 仅权重量化 (GPTQ/AWQ) | 需量化权重 |

---

## 四、Qwen 多模态模型问题

### 4.1 Qwen2-VL / Qwen2.5-VL 图像处理参数

**现象：**
```
ValueError: Image processing failed or unexpected error
```

**原因：** Qwen-VL 系列需要通过 `mm_processor_kwargs` 传递图像处理器参数。

**解决：**
```python
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    mm_processor_kwargs={"num_crops": 4},  # 控制图像裁剪数
    limit_mm_per_prompt={"image": 5},       # 限制每请求最大图片数
)
```

### 4.2 Qwen-VL 高分辨率图片 OOM

**现象：** 输入高分辨率图片时 GPU OOM。

**原因：** Qwen-VL 的动态分辨率机制会将大图分解为多个 tile，每个 tile 增加大量 token。

**解决：**
```python
# 方法 1：限制最大分辨率
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    mm_processor_kwargs={"num_crops": 2},  # 减少裁剪数
    max_model_len=4096,                    # 限制总长度
)

# 方法 2：使用较小图片
# 在传入前将图片缩放到合理尺寸（如 512x512）

# 方法 3：使用 FP8 量化节省显存
llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    quantization="fp8",
)
```

### 4.3 Qwen2-Audio 音频处理

**解决：**
```python
llm = LLM(
    model="Qwen/Qwen2-Audio-7B-Instruct",
    limit_mm_per_prompt={"audio": 3},  # 限制音频数
)
```

---

## 五、Qwen3.5 混合模型问题

### 5.1 Qwen3.5 是混合注意力+Mamba 模型

**现象：** Qwen3.5 在 vLLM 中启用时有额外日志输出，且某些配置自动调整。

**原因：** Qwen3.5 使用 Mamba 层 + 注意力层的混合架构（`IsHybrid` 接口），vLLM 会自动应用 `MambaModelConfig` 配置：
- 默认启用 `FULL_AND_PIECEWISE` CUDA Graph 模式
- 自动设置 `mamba_ssm_cache_dtype`（从 HF config 的 `mamba_ssm_dtype` 读取）
- 前缀缓存时自动设置 `mamba_cache_mode`

**解决：** 通常无需手动干预，vLLM 自动处理。但需注意：

```python
# 如果手动指定了 mamba_ssm_cache_dtype 且与模型 config 不一致，会收到警告
llm = LLM(
    model="Qwen/Qwen3.5-7B",
    mamba_ssm_cache_dtype="float32",  # 可能与模型 config 冲突
)

# 建议使用 auto（默认）
llm = LLM(model="Qwen/Qwen3.5-7B")
```

### 5.2 Qwen3.5 不支持 `calculate_kv_scales`

**现象：**
```
WARNING: Disabling calculate_kv_scales for hybrid model 'Qwen3.5-7B'. 
Hybrid models with recurrent layers produce unreliable KV cache scales.
```

**原因：** 混合模型中 Mamba/SSM 层的循环状态在标定阶段未初始化，导致 KV Cache 缩放值不可靠。

**解决：** 无需处理，vLLM 自动禁用。

---

## 六、NVIDIA GPU 特定问题

### 6.1 Qwen2-72B 在 A100 80G 单卡 OOM

**现象：** `torch.cuda.OutOfMemoryError`

**解决：**
```python
# 方法 1：使用 2 卡张量并行
llm = LLM(model="Qwen/Qwen2.5-72B-Instruct", tensor_parallel_size=2)

# 方法 2：使用 FP8 量化后单卡
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct-FP8",  # 使用官方 FP8 版本
    # 或
    # model="Qwen/Qwen2.5-72B-Instruct",
    # quantization="fp8",
)

# 方法 3：使用 AWQ/GPTQ 量化版本
llm = LLM(model="Qwen/Qwen2.5-72B-Instruct-AWQ")

# 方法 4：限制模型长度 + CPU 卸载
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    max_model_len=4096,
    cpu_offload_gb=10,
)
```

### 6.2 Qwen3 系列在 V100 上不支持 BF16

**现象：**
```
Your device 'NVIDIA V100' doesn't support bfloat16. Falling back to float16.
```

**原因：** V100 (Compute Capability 7.0) 不原生支持 BF16。

**解决：**
```python
# V100 需显式指定 float16
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="float16")
```

> **注意：** GLM4 模型明确不支持 float16（数值不稳定），在 V100 上无法运行。

### 6.3 Qwen3 思维链（Thinking Mode）输出

**现象：** Qwen3 默认启用思维链模式，输出包含 `<think>...</think>` 内容。

**解决：**
```python
# 方法 1：在 SamplingParams 中设置 stop
sampling_params = SamplingParams(
    max_tokens=1024,
    stop=["<|im_end|>"],
)

# 方法 2：通过 chat_template 控制
# 在 system prompt 中加入："/no_think" 可关闭思维链

# 方法 3：使用 reasoning_parser（API Server 模式）
# vllm serve Qwen/Qwen3-8B --reasoning-parser deepseek_r1
```

---

## 七、昇腾 910B 特定问题

### 7.1 vLLM 对昇腾 910B 的支持状态

> **重要提示：** 截至vLLM 0.21.0，官方代码中 **没有** 独立的 NPU/Ascend 平台实现。`vllm/platforms/` 目录下仅有 `cuda.py`、`rocm.py`、`xpu.py`、`cpu.py`、`tpu.py`。

昇腾 910B 的 vLLM 适配通常依赖以下途径：
1. **华为 vLLM 分支**：华为维护的 vLLM fork，包含 NPU 适配
2. **社区适配**：基于 vLLM 的 NPU 后端实现

### 7.2 昇腾 910B 通用排错

**环境变量设置：**
```bash
# 指定 NPU 可见设备
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# Ray 框架下 NPU 设备设置
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# 内存相关
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # NPU 上推荐 spawn
```

**常见问题：**

| 问题 | 原因 | 解决 |
|------|------|------|
| CUDA kernel 编译失败 | NPU 不支持 CUDA | 使用华为适配分支，或使用 CPU fallback |
| BF16 数值异常 | 部分 910B 固件版本 BF16 精度不足 | 升级固件，或使用 FP32 |
| NCCL 初始化失败 | NPU 使用 HCCL 而非 NCCL | 确保使用华为适配版本的通信库 |
| FlashAttention 不可用 | NPU 无原生 FA | 使用 NPU 专用注意力后端 |

### 7.3 Qwen 在 910B 上的部署建议

```python
# 910B 上推荐配置
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",              # 910B 对 FP16 支持好于 BF16
    enforce_eager=True,           # 禁用 CUDA Graph（NPU 不支持）
    device="auto",                # 自动检测设备
    gpu_memory_utilization=0.85,  # 910B 64GB 显存，留够余量
    max_model_len=4096,           # 保守设置长度
)
```

**910B 上 Qwen MoE 模型：**
- Qwen2-57B-A14B：需要至少 2 张 910B（64GB）
- Qwen3-235B-A22B：需要至少 4-8 张 910B
- 建议使用 FP8 量化版本减小显存占用

### 7.4 910B 上多模态模型

Qwen-VL 系列在 910B 上需要特别注意：
- 图像编码器（ViT）的算子兼容性
- 动态分辨率 tile 切分可能在 NPU 上有性能差异
- 建议先测试纯文本推理是否正常，再测试多模态功能

---

## 八、Qwen 推荐部署配置速查

| 模型 | 参数量 | 最低 GPU 配置 | 推荐 GPU 配置 | 关键参数 |
|------|--------|-------------|-------------|----------|
| Qwen2.5-7B | 7B | 1x 16GB | 1x 24GB | `dtype="bfloat16"` |
| Qwen2.5-14B | 14B | 1x 24GB | 1x 48GB | `dtype="bfloat16"` |
| Qwen2.5-32B | 32B | 2x 24GB | 2x 48GB | `tp=2, dtype="bfloat16"` |
| Qwen2.5-72B | 72B | 2x 80GB | 4x 80GB | `tp=4, dtype="bfloat16"` 或 `tp=2, fp8` |
| Qwen2-57B-A14B | 57B | 2x 48GB | 2x 80GB | `tp=2, dtype="bfloat16"` |
| Qwen3-8B | 8B | 1x 16GB | 1x 24GB | thinking mode 注意 stop tokens |
| Qwen3-32B | 32B | 2x 24GB | 2x 48GB | `tp=2, dtype="bfloat16"` |
| Qwen3-235B-A22B | 235B | 4x 80GB | 8x 80GB | `tp=8, enable_expert_parallel=True` |
| Qwen2.5-VL-7B | 7B | 1x 24GB | 1x 48GB | `mm_processor_kwargs={"num_crops": 4}` |
| Qwen3.5-7B | 7B | 1x 24GB | 1x 48GB | 混合 Mamba+Attention 模型 |

---

## 九、环境变量速查

| 环境变量 | 作用 | 常用值 |
|----------|------|--------|
| `VLLM_MLA_DISABLE` | 禁用 MLA 注意力优化 | `1`（调试时） |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | 允许设置超过模型 config 的长度 | `1` |
| `VLLM_WORKER_MULTIPROC_METHOD` | Worker 多进程启动方式 | `spawn`（NPU 推荐） |
| `CUDA_VISIBLE_DEVICES` | 指定可见 GPU | `0,1,2,3` |
| `ASCEND_RT_VISIBLE_DEVICES` | 指定可见 NPU | `0,1,2,3`（910B） |
