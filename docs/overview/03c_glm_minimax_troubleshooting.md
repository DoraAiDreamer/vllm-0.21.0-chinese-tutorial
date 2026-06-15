# GLM / MiniMax 系列模型部署常见问题与排错

> 基于 vllm-0.21.0 版本

本文档整理 GLM（智谱 AI）和 MiniMax 系列模型在 vLLM 上部署时遇到的常见问题、踩坑记录及解决方案，覆盖 NVIDIA GPU 和华为昇腾 910B 两种硬件平台。

---

## 一、模型家族概览

### 1.1 GLM 系列

| 模型 | 架构名 | 类型 | 文件 |
|------|--------|------|------|
| ChatGLM | `ChatGLMForCausalLM` | Decoder-only | `chatglm.py` |
| GLM-4 | `GlmForCausalLM` | Decoder-only | `glm.py` |
| GLM-4V | `Glm4VForCausalLM` | 视觉多模态 | `glm4v.py` |
| GLM-4-1V | `Glm4_1VForCausalLM` | 视觉多模态 | `glm4_1v.py` |
| GLM-4-MoE | `Glm4MoeForCausalLM` | MoE | `glm4_moe.py` |
| GLM-4-MoE-Lite | `Glm4MoeLiteForCausalLM` | MoE Lite | `glm4_moe_lite.py` |
| GLM-4-MoE-MTP | `Glm4MoeMTPForCausalLM` | MoE+MTP | `glm4_moe_mtp.py` |
| GLM-4-ASR | `GLM4ASRForCausalLM` | 语音识别 | `glmasr.py` |
| GLM-OCR | `GLM4OCRForCausalLM` | OCR | `glm_ocr.py` |
| Midashi-GLM | `MidasShengLmForCausalLM` | 混合 | `midashenglm.py` |

### 1.2 MiniMax 系列

| 模型 | 架构名 | 类型 | 文件 |
|------|--------|------|------|
| MiniMax-M1 | `MiniMaxText01ForCausalLM` | 混合注意力+Mamba | `minimax_text_01.py` |
| MiniMax-M2 | `MiniMaxM2ForCausalLM` | MoE | `minimax_m2.py` |
| MiniMax-VL-01 | `MiniMaxVL01ForConditionalGeneration` | 视觉多模态 | `minimax_vl_01.py` |

---

## 二、GLM 系列通用问题

### 2.1 GLM-4 不支持 float16（关键坑）

**现象：**
```
ValueError: The model type 'glm4' does not support float16.
Reason: Numerical instability. Please use bfloat16 or float32 instead.
```

**原因：** GLM-4 模型架构在 float16 精度下存在数值不稳定问题，会导致生成结果完全错误。vLLM 0.21.0 在 `_FLOAT16_NOT_SUPPORTED_MODELS` 中显式列出了 `glm4`。

**解决：**
```python
# 必须使用 bfloat16 或 float32
llm = LLM(
    model="THUDM/glm-4-9b-chat",
    dtype="bfloat16",  # 推荐
    # dtype="float32",  # 备选（显存占用更大）
)

# 如果 dtype="auto" 自动选择了 float16，会报错
# 必须显式指定
```

```bash
# CLI 方式
vllm serve THUDM/glm-4-9b-chat --dtype bfloat16
```

**重要：** 如果你的 GPU 不支持 BF16（如 V100），GLM-4 模型无法在 vLLM 上运行。

### 2.2 ChatGLM (1.0/2.0/3.0) 的自定义架构

**现象：**
```
ValueError: Model 'THUDM/chatglm3-6b' architecture not supported
```

**原因：** ChatGLM 1.0/2.0/3.0 使用非标准架构名，需要 `trust_remote_code=True`。

**解决：**
```python
# ChatGLM 3.0
llm = LLM(
    model="THUDM/chatglm3-6b",
    trust_remote_code=True,
    dtype="float16",  # ChatGLM 3.0 支持 float16
)

# ChatGLM 2.0
llm = LLM(
    model="THUDM/chatglm2-6b",
    trust_remote_code=True,
)

# ChatGLM 1.0
llm = LLM(
    model="THUDM/chatglm-6b",
    trust_remote_code=True,
)
```

### 2.3 GLM-4 的 Chat Template

**现象：** 对话格式不正确，模型回复混乱。

**原因：** GLM-4 使用特殊的对话格式，与标准的 ChatML 不同。

**解决：**
```python
# GLM-4 的对话格式
messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "你好"},
]

# GLM-4 的 chat template 会自动渲染
llm = LLM(model="THUDM/glm-4-9b-chat", dtype="bfloat16")

# 使用 generate 时传入渲染后的 prompt
prompt = llm.chat(messages)
outputs = llm.generate([prompt], SamplingParams(max_tokens=512))
```

### 2.4 GLM-4 MoE 模型显存

**现象：** GLM-4-MoE 在 vLLM 上显存占用异常。

**原因：** MoE 模型所有专家权重常驻显存。

**解决：**
```python
# GLM-4-MoE（130B 总参数，激活 10B）
llm = LLM(
    model="THUDM/glm-4-moe",
    dtype="bfloat16",
    tensor_parallel_size=4,  # 4 卡 A100 80GB
    gpu_memory_utilization=0.9,
)

# GLM-4-MoE-Lite（更小版本）
llm = LLM(
    model="THUDM/glm-4-moe-lite",
    dtype="bfloat16",
    tensor_parallel_size=2,  # 2 卡 A100 80GB
)
```

---

## 三、MiniMax 系列专属问题

### 3.1 MiniMax-M1 是混合注意力+Mamba 模型

**现象：**
```
WARNING: Disabling calculate_kv_scales for hybrid model 'MiniMaxText01ForCausalLM'.
```

**原因：** MiniMax-M1 使用 `IsHybrid` 接口，结合了标准注意力层和 Mamba (SSM) 层。vLLM 自动应用 `HybridAttentionMambaModelConfig` 配置：
- 禁用 `calculate_kv_scales`（Mamba 循环状态未初始化导致标定不可靠）
- 自动启用 `FULL_AND_PIECEWISE` CUDA Graph 模式
- 前缀缓存时自动设置 `mamba_cache_mode`

**解决：** 这是自动处理，无需手动干预。但需注意：

```python
llm = LLM(
    model="MiniMaxAI/MiniMax-M1",
    dtype="bfloat16",
    # vLLM 自动处理 Mamba 相关配置
)
```

### 3.2 MiniMax-M1 前缀缓存问题

**现象：** 启用前缀缓存时出现 Mamba cache 模式警告。

**原因：** Mamba 层的缓存模式需要特殊处理。

**解决：**
```python
llm = LLM(
    model="MiniMaxAI/MiniMax-M1",
    enable_prefix_caching=True,  # 自动设置 mamba_cache_mode
)
```

日志输出：
```
Mamba cache mode is set to 'align' for MiniMaxText01ForCausalLM by default when prefix caching is enabled
```

### 3.3 MiniMax-M2 MoE 模型

**现象：** MiniMax-M2 在 vLLM 上加载失败或性能低。

**原因：** M2 是 MoE 架构，需要张量并行。

**解决：**
```python
llm = LLM(
    model="MiniMaxAI/MiniMax-M2",
    dtype="bfloat16",
    tensor_parallel_size=4,  # 根据显存调整
)
```

### 3.4 MiniMax 模型权重名称映射

**现象：**
```
KeyError: 'model.layers.0.self_attn.qkv_proj.weight'
```

**原因：** MiniMax 模型的权重名称与 vLLM 标准命名不同，需要自定义权重映射。

**解决：** vLLM 0.21.0 的 [minimax_text_01.py](vllm/model_executor/models/minimax_text_01.py) 已内置 `replace_weight_name` 函数处理权重映射，一般无需手动干预。

---

## 四、GLM 多模态模型问题

### 4.1 GLM-4V 视觉多模态

**现象：** GLM-4V 图片处理失败。

**解决：**
```python
llm = LLM(
    model="THUDM/glm-4v-9b",
    dtype="bfloat16",
    mm_processor_kwargs={"num_crops": 4},
    limit_mm_per_prompt={"image": 5},
)
```

### 4.2 GLM-4-OCR

**现象：** OCR 输出不完整或格式错误。

**解决：**
```python
llm = LLM(
    model="THUDM/glm-4-ocr",
    dtype="bfloat16",
)
```

---

## 五、NVIDIA GPU 特定问题

### 5.1 GLM-4 在 V100 上不可用

**现象：**
```
Your device 'NVIDIA V100' doesn't support bfloat16. Falling back to float16.
ValueError: The model type 'glm4' does not support float16.
```

**原因：** V100 不支持 BF16，而 GLM-4 不支持 FP16。

**解决：** 无解。V100 上无法运行 GLM-4 模型。考虑：
- 使用 GLM-3 替代（支持 FP16）
- 迁移到支持 BF16 的 GPU（A100/H100/RTX 4090）

### 5.2 GLM-4 在 RTX 4090 上部署

**推荐配置：**
```python
llm = LLM(
    model="THUDM/glm-4-9b-chat",
    dtype="bfloat16",  # 4090 支持 BF16
    tensor_parallel_size=1,
    gpu_memory_utilization=0.92,
    max_model_len=8192,
)
```

### 5.3 GLM-4 在 A100 上部署

**推荐配置：**
```python
llm = LLM(
    model="THUDM/glm-4-9b-chat",
    dtype="bfloat16",
    tensor_parallel_size=1,  # 9B 单卡即可
    # 如需更大 batch
    # tensor_parallel_size=2,
)
```

### 5.4 GLM-4-MoE 在 A100 上部署

```python
llm = LLM(
    model="THUDM/glm-4-moe",
    dtype="bfloat16",
    tensor_parallel_size=4,  # 4x A100 80GB
    gpu_memory_utilization=0.9,
)
```

---

## 六、昇腾 910B 特定问题

### 6.1 GLM 在 910B 上的支持状态

> **重要提示：** 截至 vLLM 0.21.0，官方代码中 **没有** 独立的 NPU/Ascend 平台实现。昇腾 910B 的 vLLM 适配依赖华为维护的分支。

### 6.2 GLM-4 在 910B 上的排错

**常见问题：**

| 问题 | 原因 | 解决 |
|------|------|------|
| BF16 数值异常 | 部分 910B 固件版本 BF16 精度不足 | 升级固件或使用 FP32 |
| GLM-4 不支持 FP16 | GLM-4 架构限制 | 无法在 FP16 硬件上运行 |
| 注意力后端不可用 | 标准 FA 后端不支持 NPU | 使用华为适配版本 |

**910B 部署建议：**
```bash
# 环境变量
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

```python
# 910B 上 GLM 部署（需华为适配版本）
llm = LLM(
    model="THUDM/glm-4-9b-chat",
    dtype="float16",  # 910B 优先 FP16
    enforce_eager=True,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
)
```

### 6.3 MiniMax-M1 在 910B 上的问题

MiniMax-M1 是混合 Mamba+注意力模型，在 910B 上需要特别注意：
- Mamba SSM 层的算子兼容性
- 循环状态更新可能在 NPU 上有性能差异
- 建议先测试纯文本推理，再测试完整功能

---

## 七、GLM/MiniMax 推荐部署配置速查

| 模型 | 参数量 | 最低 GPU | 推荐 GPU | 关键参数 |
|------|--------|----------|----------|----------|
| ChatGLM3-6B | 6B | 1x 16GB | 1x 24GB | `trust_remote_code=True` |
| GLM-4-9B-Chat | 9B | 1x 24GB | 1x 48GB | `dtype="bfloat16"` |
| GLM-4V-9B | 9B | 1x 24GB | 1x 48GB | `dtype="bfloat16"` |
| GLM-4-MoE | 130B | 4x 80GB | 4x 80GB | `dtype="bfloat16", tp=4` |
| GLM-4-MoE-Lite | ~50B | 2x 48GB | 2x 80GB | `dtype="bfloat16", tp=2` |
| MiniMax-M1 | ~100B | 4x 80GB | 4x 80GB | 混合 Mamba+Attention |
| MiniMax-M2 | ~70B | 2x 80GB | 4x 80GB | `tp=4, dtype="bfloat16"` |

---

## 八、排错决策树

```
GLM/MiniMax 部署问题
    │
    ├── GLM-4 启动失败？
    │       ├── float16 报错 → 必须用 bfloat16
    │       ├── V100 上运行 → 不支持，换 GPU
    │       └── 对话格式错误 → 检查 chat_template
    │
    ├── ChatGLM 加载失败？
    │       └── trust_remote_code=True
    │
    ├── MiniMax-M1 有警告？
    │       ├── calculate_kv_scales 禁用 → 正常，自动处理
    │       └── mamba_cache_mode 警告 → 正常，自动处理
    │
    ├── 昇腾 910B？
    │       ├── GLM-4 不支持 FP16 → 需 BF16 硬件
    │       └── 使用华为适配分支
    │
    └── 生成质量差？
            ├── 检查 dtype → GLM-4 必须 BF16
            ├── 检查 tokenizer → 版本一致性
            └── 检查 temperature → 采样参数
```
