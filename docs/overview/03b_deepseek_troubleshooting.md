# DeepSeek 系列模型部署常见问题与排错

> 基于 vllm-0.21.0 版本

本文档整理 DeepSeek 系列模型在 vLLM 上部署时遇到的常见问题、踩坑记录及解决方案，覆盖 NVIDIA GPU 和华为昇腾 910B 两种硬件平台。

---

## 一、DeepSeek 模型家族概览

vLLM 0.21.0 支持的 DeepSeek 系列模型架构：

| 模型 | 架构名 | 类型 | 文件 |
|------|--------|------|------|
| DeepSeek-V2 | `DeepseekV2ForCausalLM` | MLA+MoE | `deepseek_v2.py` |
| DeepSeek-V3 | `DeepseekV3ForCausalLM` | MLA+MoE | `deepseek_v2.py` |
| DeepSeek-V3.2 | `DeepseekV32ForCausalLM` | MLA+MoE | `deepseek_v2.py` |
| DeepSeek-V4 | `DeepseekV4ForCausalLM` | MLA+MoE+FP8 | `deepseek_v4.py` |
| DeepSeek-Eagle | `DeepSeekEagleForCausalLM` | 蒸馏模型 | `deepseek_eagle.py` |
| DeepSeek-Eagle3 | `Eagle3DeepseekV3ForCausalLM` | 蒸馏模型 | `deepseek_eagle3.py` |
| DeepSeek-VL2 | `DeepSeek-VL2` | 视觉多模态 | `deepseek_vl2.py` |
| DeepSeek-OCR | `DeepSeekOCRForCausalLM` | OCR | `deepseek_ocr.py` |
| DeepSeek-OCR2 | `DeepSeekOCR2ForCausalLM` | OCR v2 | `deepseek_ocr2.py` |
| DeepSeek-MTP | `DeepSeekMTP` | 多 token 预测 | `deepseek_mtp.py` |

---

## 二、核心架构特点（排错基础）

DeepSeek V2/V3/V4 的核心架构特点决定了其部署方式：

1. **MLA（Multi-Latent Attention）**：将 KV Cache 压缩为 latent 向量，大幅减少 KV Cache 显存占用
   - 使用 `VLLM_MLA_DISABLE=0`（默认）启用 MLA 优化
   - 使用 `VLLM_MLA_DISABLE=1` 禁用 MLA 优化（调试用）

2. **MoE（Mixture of Experts）**：稀疏激活专家
   - 需要张量并行 + 可选专家并行
   - 所有专家权重常驻显存

3. **FP8 量化**（V4 特有）：
   - V4 默认使用 FP8 权重 + MXFP4 专家量化
   - 需要支持 FP8 计算的 GPU（H100+ 或 A100）

---

## 三、通用问题

### 3.1 MLA 注意力问题

**现象 1：** 生成结果与 HF 推理不一致

**原因：** MLA 的 latent 压缩可能导致精度损失，尤其在小 batch 或特定 attention head 配置下。

**解决：**
```python
# 方法 1：禁用 MLA 优化（牺牲性能换取正确性）
import os
os.environ["VLLM_MLA_DISABLE"] = "1"

llm = LLM(model="deepseek-ai/DeepSeek-V3")

# 方法 2：使用 FP8 KV Cache 提高精度
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    kv_cache_dtype="fp8",
)

# 方法 3：指定 FlashInfer MLA 后端
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    attention_config={
        "backend": "flashinfer",
        "mla_backend": "flashinfer_mla",  # 使用 FlashInfer 原生 MLA
    },
)
```

**现象 2：** 性能异常低

**原因：** 默认 MLA 后端可能未选择最优实现。

**解决：**
```python
# 检查当前后端
llm = LLM(model="deepseek-ai/DeepSeek-V3")  # 查看启动日志

# 强制指定 FlashInfer MLA
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    attention_config={
        "backend": "flashinfer",
    },
)
```

### 3.2 显存不足（OOM）

**现象：**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...
```

**原因：** DeepSeek V2/V3 参数量大（671B 总参数，37B 激活），即使 MoE 稀疏激活，所有权重仍需加载。

**解决：**
```python
# 方法 1：张量并行（必须）
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,  # V3 建议 8 卡 A100 80GB
)

# 方法 2：专家并行（大规模推荐）
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    enable_expert_parallel=True,  # 专家权重分片
)

# 方法 3：FP8 量化（V2/V3 可用）
llm = LLM(
    model="deepseek-ai/DeepSeek-V2",
    quantization="fp8",
    tensor_parallel_size=4,
)

# 方法 4：AWQ 量化版本
llm = LLM(
    model="deepseek-ai/DeepSeek-V2-AWQ",  # 使用 AWQ 量化版
    tensor_parallel_size=4,
)

# 方法 5：GPTQ 量化版本
llm = LLM(
    model="deepseek-ai/DeepSeek-V2-GPTQ",
    quantization="gptq",
    tensor_parallel_size=4,
)
```

### 3.3 DeepSeek-V3.2 的 BF16 KV Cache 问题

**现象：**
```
WARNING: Using bfloat16 kv-cache for DeepSeekV3.2
```

**原因：** DeepSeek-V3.2 在 vLLM 中自动将 KV Cache dtype 从默认改为 `bfloat16`（而非 FP16），因为 FP16 会导致数值不稳定。

**解决：** 这是 vLLM 的自动处理，无需手动干预。但需注意：
- 如果你的 GPU 不支持 BF16（如 V100），此模型无法运行
- BF16 KV Cache 比 FP16 占用更多显存（BF16 和 FP16 都是 2 字节，但 BF16 范围更大，需要更多对齐）

```python
# 显式指定（通常不需要）
llm = LLM(
    model="deepseek-ai/DeepSeek-V3.2",
    kv_cache_dtype="bfloat16",  # 自动设置为 bfloat16
)
```

### 3.4 生成质量下降

**现象：** DeepSeek V2/V3 在 vLLM 上生成的文本质量不如 HuggingFace 推理。

**原因排查：**

| 可能原因 | 排查方法 | 解决 |
|----------|----------|------|
| MLA 数值精度 | 对比 `VLLM_MLA_DISABLE=1` 和默认 | 如禁用后一致，说明 MLA 精度问题 |
| CUDA Graph 编译 | 检查 `enforce_eager` 设置 | 设 `enforce_eager=True` 对比 |
| KV Cache 块大小 | 检查 block_size | 设 `block_size=16`（默认） |
| Attention 后端 | 检查自动选择的后端 | 显式指定 `flashinfer` |
| Tokenizer 差异 | 对比 tokenizer 输出 | 确保使用相同 tokenizer 版本 |

```python
# 排查用：禁用所有优化
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    enforce_eager=True,           # 禁用 CUDA Graph
    tensor_parallel_size=1,        # 单卡调试
)

# 对比 MLA 禁用/启用
# 先设 VLLM_MLA_DISABLE=0（默认），对比 VLLM_MLA_DISABLE=1
```

---

## 四、DeepSeek V4 专属问题

### 4.1 V4 需要 FP8 计算硬件

**现象：**
```
RuntimeError: FP8 is not supported on this device
```

**原因：** DeepSeek V4 使用 FP8 量化权重 + MXFP4 专家量化，需要 GPU 支持 FP8 计算。

**支持 FP8 的 GPU：**
- NVIDIA H100/H200/H800（原生 FP8 Tensor Core）
- NVIDIA A100/A800（FP8 通过 Tensor Core 支持，性能较好）
- NVIDIA L40S（支持 FP8）
- NVIDIA B100/B200（原生 FP8）

**不支持 FP8 的 GPU：**
- V100、T4、RTX 3090、RTX 4090 等

**解决：**
```python
# 方案 1：使用 FP8 量化版本（推荐，在支持 FP8 的 GPU 上）
llm = LLM(
    model="deepseek-ai/DeepSeek-V4-Flash",  # MXFP4 专家版本
    tensor_parallel_size=8,
)

# 方案 2：使用 FP8 量化版本（Base 版）
llm = LLM(
    model="deepseek-ai/DeepSeek-V4-Flash-Base",  # FP8 专家版本
    tensor_parallel_size=8,
)

# 方案 3：在 A100 上强制使用 FP8 Marlin
import os
os.environ["VLLM_FORCE_FP8_MARLIN"] = "1"

llm = LLM(
    model="deepseek-ai/DeepSeek-V4-Flash",
    tensor_parallel_size=4,
)
```

### 4.2 V4 的 Expert Dtype 自动识别

**现象：**
```
INFO: DeepSeek V4 expert_dtype resolved to 'fp4'
```

**原因：** V4 支持两种专家量化：
- `expert_dtype="fp4"`（MXFP4，如 DeepSeek-V4-Flash）
- `expert_dtype="fp8"`（如 DeepSeek-V4-Flash-Base）

vLLM 会自动从 HF config 中识别，无需手动指定。

**注意：** 如果手动指定了错误的 `expert_dtype`，会报错：
```
ValueError: Unsupported DeepSeek V4 expert_dtype='fp16'; 
expected one of ('fp4', 'fp8').
```

### 4.3 V4 的量化配置

**现象：** V4 模型在 vLLM 中加载时量化方式需要特殊处理。

**原因：** V4 的量化配置在 HF config 中为 `fp8`，但 vLLM 需要映射到内部的 `deepseek_v4_fp8` 量化方法。

**解决：** vLLM 0.21.0 已自动处理此映射（在 `DeepseekV4ForCausalLMConfig.verify_and_update_model_config` 中），无需手动干预。

```python
# 自动处理，无需手动指定 quantization
llm = LLM(model="deepseek-ai/DeepSeek-V4-Flash")
```

### 4.4 V4 显存需求

| 配置 | 显存需求 | GPU 配置建议 |
|------|----------|-------------|
| DeepSeek-V4-Flash (671B) | ~350GB | 8x A100 80GB (FP8) |
| DeepSeek-V4-Flash-Base (671B) | ~400GB | 8x A100 80GB (FP8) |

```python
# V4 推荐部署配置
llm = LLM(
    model="deepseek-ai/DeepSeek-V4-Flash",
    tensor_parallel_size=8,
    enable_expert_parallel=True,  # 专家并行降低单卡显存
    kv_cache_dtype="auto",        # 自动选择
    moe_backend="deep_gemm",      # FP8 专用后端
)
```

---

## 五、DeepSeek MoE 模型问题

### 5.1 专家并行（Expert Parallelism）配置

**现象：** V2/V3 多卡部署时显存利用率低，部分卡 OOM。

**原因：** 默认仅使用张量并行，所有专家权重在每张卡上复制。

**解决：**
```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=4,
    enable_expert_parallel=True,  # 专家权重分片到不同卡
)
```

```bash
# CLI
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel
```

**注意事项：**
- `enable_expert_parallel=True` 需要 NCCL 通信库正常
- 专家并行与流水线并行不兼容
- 专家并行适合专家数远大于 TP size 的场景

### 5.2 MoE Backend 选择

**现象：** MoE 推理速度慢，或出现 kernel 报错。

**解决：**
```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    moe_backend="triton",           # 通用兼容
    # moe_backend="flashinfer_cutlass",  # 高性能（需 FlashInfer）
    # moe_backend="cutlass",           # CUTLASS 后端
)
```

### 5.3 DeepSeek-V1 模型（已废弃）

DeepSeek-V1 使用标准 MHA 而非 MLA，在 vLLM 0.21.0 中仍受支持但已不推荐：
- V1 的显存占用远高于 V2/V3
- 建议使用 V2 或 V3 版本替代

---

## 六、DeepSeek 推测解码问题

### 6.1 Eagle 蒸馏模型

**现象：** 使用 Eagle 作为草稿模型加速 V3 推理时性能异常。

**解决：**
```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    speculative_model="deepseek-ai/DeepSeek-Eagle3",  # Eagle3 蒸馏模型
    num_speculative_tokens=8,
    tensor_parallel_size=8,
)
```

```bash
# CLI
vllm serve deepseek-ai/DeepSeek-V3 \
    --speculative-model deepseek-ai/DeepSeek-Eagle3 \
    --num-speculative-tokens 8 \
    --tensor-parallel-size 8
```

**注意事项：**
- Eagle3 支持 V3 和 Qwen3-VL 的蒸馏
- `num_speculative_tokens` 建议 4-8
- 推测解码可能降低生成质量（需调整 temperature）

---

## 七、NVIDIA GPU 特定问题

### 7.1 DeepSeek V3 在 A100 80GB 上的部署

**推荐配置：**
```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,       # 8 卡 A100 80GB
    enable_expert_parallel=True,  # 专家并行
    kv_cache_dtype="auto",        # 自动选择
    gpu_memory_utilization=0.92,
)
```

### 7.2 DeepSeek V3 在 H100 80GB 上的部署

**推荐配置：**
```python
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    gpu_memory_utilization=0.95,  # H100 内存效率更高，可更高
    moe_backend="deep_gemm",      # H100 专用 FP8 后端
)
```

### 7.3 DeepSeek V3 在 RTX 4090 上的部署

**限制：** RTX 4090 不支持 FP8，且显存仅 24GB。

**可行方案：**
```python
# 方案 1：使用 AWQ 量化版本（需 2-4 卡）
llm = LLM(
    model="deepseek-ai/DeepSeek-V2-AWQ",
    quantization="awq",
    tensor_parallel_size=4,  # 4 张 4090
)

# 方案 2：使用 GPTQ 量化版本
llm = LLM(
    model="deepseek-ai/DeepSeek-V2-GPTQ",
    quantization="gptq",
    tensor_parallel_size=4,
)

# 方案 3：仅支持小模型（DeepSeek-R1-Distill-Llama-8B）
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    tensor_parallel_size=1,
)
```

### 7.4 DeepSeek V3 的 FlashInfer 依赖

**现象：**
```
ImportError: cannot import name 'flashinfer'
```

**解决：**
```bash
# 安装 FlashInfer
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.5/
```

```python
# 使用 FlashInfer 后端
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    attention_config={"backend": "flashinfer"},
)
```

---

## 八、昇腾 910B 特定问题

### 8.1 DeepSeek V3 在 910B 上的支持状态

> **重要提示：** DeepSeek V3/V4 使用 MLA 注意力 + MoE 架构，其 Triton kernel 和 FlashInfer 后端在昇腾 910B 上 **不受支持**。

**可行方案：**
1. 使用华为 vLLM 适配分支
2. 使用量化版本 + CPU fallback
3. 迁移到 NVIDIA GPU 部署

### 8.2 910B 上 DeepSeek 通用排错

```bash
# 环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

**常见问题：**

| 问题 | 原因 | 解决 |
|------|------|------|
| Triton kernel 不可用 | 910B 不支持 Triton | 使用华为适配版本 |
| MLA 后端不可用 | FlashInfer 不支持 NPU | 禁用 MLA 或等待适配 |
| MoE kernel 报错 | DeepGEMM 不支持 NPU | 使用 `moe_backend="triton"` |
| FP8 不可用 | 910B FP8 支持有限 | 使用 FP16/BF16 |

### 8.3 910B 上 DeepSeek 部署建议

```python
# 910B 上仅建议部署小模型
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    dtype="float16",
    enforce_eager=True,
    tensor_parallel_size=2,  # 2 张 910B
    gpu_memory_utilization=0.85,
)
```

---

## 九、DeepSeek 推荐部署配置速查

| 模型 | 总参 | 激活参 | 最低 GPU | 推荐 GPU | 关键参数 |
|------|------|--------|----------|----------|----------|
| DeepSeek-V2 | 16B | 2.2B | 1x 24GB | 1x 48GB | `dtype="bfloat16"` |
| DeepSeek-V2-671B | 671B | 24B | 8x 80GB | 8x 80GB | `tp=8, ep=True` |
| DeepSeek-V3 | 671B | 37B | 8x 80GB | 8x 80GB | `tp=8, ep=True` |
| DeepSeek-V3.2 | 253B | 16B | 4x 80GB | 4x 80GB | `tp=4, ep=True` |
| DeepSeek-V4-Flash | 671B | 21B | 8x 80GB | 8x H100 | `tp=8, fp8, ep=True` |
| DeepSeek-R1-Distill-7B | 7B | 7B | 1x 16GB | 1x 24GB | 小模型 |
| DeepSeek-R1-Distill-8B | 8B | 8B | 1x 16GB | 1x 24GB | 小模型 |
| DeepSeek-R1-Distill-14B | 14B | 14B | 1x 24GB | 1x 48GB | 小模型 |
| DeepSeek-R1-Distill-32B | 32B | 32B | 2x 24GB | 2x 48GB | `tp=2` |
| DeepSeek-R1-Distill-70B | 70B | 70B | 2x 80GB | 4x 80GB | `tp=4` |

---

## 十、环境变量速查

| 环境变量 | 作用 | 常用值 |
|----------|------|--------|
| `VLLM_MLA_DISABLE` | 禁用 MLA 优化（调试用） | `0`（默认）/ `1` |
| `VLLM_FORCE_FP8_MARLIN` | 强制使用 FP8 Marlin | `1` |
| `VLLM_WORKER_MULTIPROC_METHOD` | Worker 启动方式 | `spawn`（NPU 推荐） |
| `VLLM_MLA_DISABLE` | 禁用 MLA 优化 | `1`（排查精度问题） |

---

## 十一、排错决策树

```
DeepSeek 模型部署问题
    │
    ├── 启动失败？
    │       ├── FP8 报错 → 检查 GPU 是否支持 FP8（H100/A100 支持，4090 不支持）
    │       ├── OOM → 增加 TP size 或使用量化版本
    │       └── 依赖缺失 → 安装 FlashInfer
    │
    ├── 生成质量差？
    │       ├── 对比 HF 推理 → 如一致，模型本身问题
    │       ├── 对比 VLLM_MLA_DISABLE=1 → 如一致，MLA 精度问题
    │       ├── 对比 enforce_eager=True → 如一致，CUDA Graph 问题
    │       └── 检查 temperature/top_p → 采样参数问题
    │
    ├── 性能低？
    │       ├── 检查 MoE backend → 尝试 deep_gemm / flashinfer_cutlass
    │       ├── 检查 CUDA Graph → 检查 cudagraph_capture_sizes
    │       ├── 检查 batch size → 增大 max_num_batched_tokens
    │       └── 检查专家并行 → 启用 enable_expert_parallel
    │
    └── 昇腾 910B？
            ├── MLA 不可用 → 使用华为适配分支
            ├── MoE kernel 报错 → 使用 triton 后端
            └── FP8 不可用 → 使用 FP16 量化版本
```
