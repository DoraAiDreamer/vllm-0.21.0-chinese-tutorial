# LLM 启动与部署指南

> 基于 vllm-0.21.0 版本

本文档提供 vLLM 的完整使用教程，涵盖 LLM 启动、参数传递、重要参数含义，以及陌生模型部署方法。

---

## 一、LLM 启动方式

vLLM 提供三种启动方式：**离线推理（Offline Inference）**、**在线服务（Online Serving）**、**命令行接口（CLI）**。

### 1.1 离线推理 — `LLM` 类

适合批量推理、脚本化调用、Jupyter Notebook 等场景。

```python
from vllm import LLM, SamplingParams

# 1. 创建 LLM 实例（加载模型）
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,          # 使用 2 张 GPU
    gpu_memory_utilization=0.9,      # GPU 内存利用率 90%
    dtype="auto",                    # 自动选择数据类型
)

# 2. 定义采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop=["<|eot_id|>", "<|eom_id|>"],
)

# 3. 生成文本
outputs = llm.generate("Hello, how are you?", sampling_params)

# 4. 输出结果
for output in outputs:
    print(output.outputs[0].text)
```

**流式生成：**

```python
for output in llm.generate("Hello, how are you?", sampling_params, stream=True):
    for request in output:
        print(request.outputs[0].text, end="", flush=True)
print()
```

### 1.2 在线服务 — OpenAI 兼容 API Server

适合生产部署、API 服务、与第三方应用集成。

**方式一：Python API 启动**

```python
from vllm.entrypoints.api_server import start_engine

# 异步引擎（推荐用于服务）
engine = start_engine(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.9,
    host="0.0.0.0",
    port=8000,
)
```

**方式二：CLI 命令启动（推荐）**

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 8000
```

启动后可通过 OpenAI 兼容接口访问：

```bash
# 聊天补全
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'

# 文本补全
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Once upon a time,",
    "temperature": 0.7,
    "max_tokens": 64
  }'
```

**方式三：YAML 配置文件**

```yaml
# config.yaml
model: meta-llama/Llama-3.1-8B-Instruct
host: "0.0.0.0"
port: 8000
tensor-parallel-size: 2
gpu-memory-utilization: 0.9
dtype: auto
```

```bash
vllm serve --config config.yaml
```

### 1.3 命令行接口 — `vllm` CLI

适合快速测试和单条请求。

```bash
# 聊天模式
vllm chat meta-llama/Llama-3.1-8B-Instruct

# 补全模式
vllm complete meta-llama/Llama-3.1-8B-Instruct \
  --prompt "The future of AI is" \
  --max-tokens 100 \
  --temperature 0.7
```

---

## 二、参数传递链路

vLLM 的参数传递涉及三层结构：

```
用户代码 / CLI
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Layer 1: LLM 类 / AsyncEngine / CLI 参数       │
│  - model, tokenizer, tensor_parallel_size, ...  │
│  - 这些参数最终合并为 VllmConfig                  │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  Layer 2: EngineArgs → VllmConfig               │
│  - EngineArgs 将用户参数解析为各子配置            │
│  - VllmConfig 聚合所有子配置（20+ 个）            │
└──────────────┬──────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────┐
│  Layer 3: 各子配置类（Config）                   │
│  - ModelConfig     (模型路径、类型、分词器)       │
│  - CacheConfig     (KV Cache 设置)               │
│  - ParallelConfig  (并行策略)                    │
│  - SchedulerConfig (调度参数)                    │
│  - CompilationConfig (编译优化)                  │
│  - LoRAConfig      (LoRA 设置)                   │
│  - SpeculativeConfig (推测解码)                  │
│  - ... (还有 10+ 个子配置)                       │
└─────────────────────────────────────────────────┘
```

### 2.1 参数传递代码路径

```
LLM.__init__()
    │
    ├── 收集所有参数（含 **kwargs）
    │
    ▼
EngineArgs(**kwargs)
    │  将 argparse 风格的参数解析为各子配置
    │
    ▼
VllmConfig  (由 EngineArgs.create_engine_config() 构建)
    │
    ├── ModelConfig    ← model, tokenizer, dtype, trust_remote_code, ...
    ├── CacheConfig    ← gpu_memory_utilization, kv_cache_memory_bytes, ...
    ├── ParallelConfig ← tensor_parallel_size, pipeline_parallel_size, ...
    ├── SchedulerConfig← max_num_batched_tokens, max_num_seqs, ...
    ├── LoadConfig     ← load_format, download_dir, ...
    ├── CompilationConfig ← compilation_config, ...
    ├── LoRAConfig     ← enable_lora, ...
    ├── SpeculativeConfig ← speculative_model, ...
    ├── AttentionConfig ← attention_backend, ...
    └── ... (其他子配置)
    │
    ▼
LLMEngine(VllmConfig) → EngineCore(VllmConfig)
```

### 2.2 在线服务的参数传递

```
CLI args / YAML config
    │
    ▼
AsyncEngineArgs  (继承自 EngineArgs，额外支持异步相关参数)
    │
    ▼
VllmConfig
    │
    ▼
AsyncLLMEngine → AsyncLLM → EngineCore
```

**优先级规则：** `命令行参数 > YAML 配置文件 > 默认值`

---

## 三、重要参数详解

### 3.1 模型加载参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | (必填) | HuggingFace 模型名称或本地路径 |
| `tokenizer` | 同 model | HuggingFace tokenizer 名称或路径 |
| `tokenizer_mode` | `"auto"` | `"auto"` 使用 fast tokenizer，`"slow"` 始终使用慢速 |
| `trust_remote_code` | `False` | 是否信任 HuggingFace 远程代码（加载自定义模型时设为 `True`） |
| `dtype` | `"auto"` | 模型数据类型：`"auto"` / `"float16"` / `"bfloat16"` / `"float32"` |
| `quantization` | `None` | 量化方法：`"awq"` / `"gptq"` / `"fp8"` / `"bitsandbytes"` / `"gguf"` 等 |
| `load_format` | `"auto"` | 权重加载格式：`"auto"` / `"safetensors"` / `"sharded_state"` / `"dummy"` / `"tensorizer"` |
| `revision` | `None` | 模型版本（branch/tag/commit） |
| `hf_token` | `None` | HuggingFace 访问令牌 |
| `hf_overrides` | `None` | HuggingFace config 覆盖（字典或回调函数） |

### 3.2 并行与硬件参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tensor_parallel_size` | `1` | 张量并行 GPU 数量 |
| `pipeline_parallel_size` | `1` | 流水线并行 GPU 数量 |
| `device` | `"auto"` | 设备类型：`"auto"` / `"cuda"` / `"rocm"` / `"xpu"` / `"cpu"` |
| `gpu_memory_utilization` | `0.9` | GPU 内存利用率（0~1），越高 KV Cache 越大，吞吐量越高 |
| `kv_cache_memory_bytes` | `None` | 手动指定 KV Cache 大小（字节），优先级高于 gpu_memory_utilization |
| `enforce_eager` | `False` | 是否强制使用 eager 模式（禁用 CUDA Graph） |
| `max_model_len` | 自动 | 最大模型长度，超过时自动缩减 |

### 3.3 调度与性能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_num_batched_tokens` | `2048` | 每批最大 token 数（含 prompt + 生成） |
| `max_num_seqs` | `256` | 每批最大请求数 |
| `block_size` | `16` | KV Cache 块大小（token 数） |
| `max_num_seqs` | `256` | 每批最大序列数 |
| `disable_custom_all_reduce` | `False` | 是否禁用自定义 all-reduce（使用 NCCL） |
| `disable_log_stats` | `False` | 是否禁用日志统计输出 |
| `log_level` | `"info"` | 日志级别 |

### 3.4 LoRA 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_lora` | `False` | 是否启用 LoRA |
| `max_loras` | `1` | 最大 LoRA 适配器数 |
| `max_lora_rank` | `16` | LoRA rank 上限 |

### 3.5 推测解码参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `speculative_model` | `None` | 草稿模型（如 `neuralmagic/Meta-Llama-3.1-8B-Instruct-AWQ`） |
| `num_speculative_tokens` | `None` | 每个草稿 token 数 |
| `speculative_pooling_mode` | `"last"` | 草稿池化模式 |

### 3.6 编译优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `compilation_config` | `None` | 编译优化配置 |
| `compile_size` | `None` | 需要编译的 batch size 列表 |

`compilation_config` 可接受的值：
- `0` — 不编译
- `1` — 编译所有支持的 batch size
- `{"inductor_compile_size": [1, 2, 4, ..., 128]}` — 指定编译的 batch sizes
- `{"pass_config": {"enable_fusion": True, "enable_activation_quant_fusion": True}}` — 详细配置

### 3.7 采样参数 (`SamplingParams`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n` | `1` | 每个请求生成 N 个输出 |
| `temperature` | `1.0` | 采样温度，越低越确定 |
| `top_p` | `1.0` | Nucleus 采样阈值 |
| `top_k` | `-1` | Top-K 采样，`-1` 表示全部 |
| `min_p` | `0.0` | Min-P 采样阈值 |
| `max_tokens` | `None` | 最大生成 token 数 |
| `stop` | `[]` | 停止字符串列表 |
| `stop_token_ids` | `[]` | 停止 token ID 列表 |
| `presence_penalty` | `0.0` | 存在惩罚（>0 鼓励新 token） |
| `frequency_penalty` | `0.0` | 频率惩罚 |
| `repetition_penalty` | `1.0` | 重复惩罚 |
| `seed` | `None` | 随机种子 |
| `logprobs` | `None` | 返回每个 token 的对数概率数 |
| `structured_outputs` | `None` | 结构化输出约束（JSON Schema / Regex / Choice） |

---

## 四、完整使用示例

### 4.1 离线批量推理

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    dtype="bfloat16",
    max_model_len=4096,
)

# 定义多个请求
prompts = [
    "Python 快速排序的代码是？",
    "解释量子计算的基本原理",
    "写一首关于春天的诗",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
    top_k=50,
)

# 批量生成
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output.outputs[0].text}")
    print("-" * 50)
```

### 4.2 聊天对话

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tokenizer_mode="auto",
    trust_remote_code=False,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# 使用 chat_template 渲染
prompt = llm.chat(messages)

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```

### 4.3 结构化输出（JSON Schema）

```python
from vllm import LLM, SamplingParams
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

sampling_params = SamplingParams(
    temperature=0.0,  # 结构化输出需要 temperature=0
    max_tokens=256,
    structured_outputs={"json_schema": json.dumps(Person.model_json_schema())},
)

outputs = llm.generate("Who is the current president of the US?", sampling_params)
print(outputs[0].outputs[0].text)
```

### 4.4 LoRA 适配器

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=4,
    max_lora_rank=8,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# 使用 LoRA 适配器生成
outputs = llm.generate(
    "Translate to French:",
    sampling_params,
    lora_request=LoRARequest("my_lora", 1, "/path/to/lora_adapter"),
)
```

### 4.5 推测解码

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_model="neuralmagic/Meta-Llama-3.1-8B-Instruct-AWQ",
    num_speculative_tokens=8,
    tensor_parallel_size=2,
)

outputs = llm.generate("Explain machine learning.", SamplingParams(max_tokens=256))
```

### 4.6 多模态模型

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    mm_processor_kwargs={"num_crops": 4},
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

# 图片 + 文本
outputs = llm.generate(
    {
        "prompt": "描述这张图片的内容",
        "multi_modal_data": {"image": "path/to/image.jpg"},
    },
    sampling_params,
)
```

---

## 五、命令行参数速查

### 5.1 `LLM` 类常用参数

```python
LLM(
    # 模型
    model="huggingface/model-name",       # HuggingFace 模型名或本地路径
    tokenizer=None,                       # tokenizer（默认同 model）
    tokenizer_mode="auto",                # "auto" | "slow"
    trust_remote_code=False,              # 信任远程代码
    dtype="auto",                         # "auto" | "float16" | "bfloat16" | "float32"
    quantization=None,                    # "awq" | "gptq" | "fp8" | "bitsandbytes" | "gguf" | ...
    load_format="auto",                   # 权重加载格式
    revision=None,                        # 模型版本
    hf_token=None,                        # HuggingFace token
    hf_overrides=None,                    # HF config 覆盖
    skip_tokenizer_init=False,            # 跳过 tokenizer 初始化
    seed=0,                               # 随机种子

    # 硬件与内存
    tensor_parallel_size=1,               # 张量并行 GPU 数
    pipeline_parallel_size=1,             # 流水线并行 GPU 数
    gpu_memory_utilization=0.9,           # GPU 内存利用率
    kv_cache_memory_bytes=None,           # KV Cache 大小（字节）
    cpu_offload_gb=0,                     # CPU 卸载大小（GB）
    enforce_eager=False,                  # 强制 eager 模式

    # 性能
    max_model_len=None,                   # 最大模型长度
    max_num_batched_tokens=2048,          # 每批最大 token 数
    max_num_seqs=256,                     # 每批最大请求数
    block_size=16,                        # KV Cache 块大小

    # 编译与注意力
    compilation_config=None,              # 编译优化配置
    attention_config=None,                # 注意力后端配置

    # LoRA
    enable_lora=False,                    # 启用 LoRA
    max_loras=1,                          # 最大 LoRA 数
    max_lora_rank=16,                     # LoRA rank

    # 多模态
    mm_processor_kwargs=None,             # 多模态处理器参数

    # 其他
    chat_template=None,                   # 聊天模板
    allowed_local_media_path="",          # 允许读取的本地媒体路径
    allowed_media_domains=None,           # 允许的媒体域名
)
```

### 5.2 `vllm serve` CLI 常用参数

```bash
vllm serve MODEL [OPTIONS]

# 模型相关
--model MODEL                      # HuggingFace 模型名或路径（必填）
--tokenizer TOKENIZER              # tokenizer 路径
--tokenizer-mode {auto,slow}       # tokenizer 模式
--trust-remote-code                # 信任远程代码
--dtype {auto,float16,bfloat16,float32,int8}
--quantization {awq,gptq,fp8,bitsandbytes,gguf,...}
--load-format {auto,safetensors,sharded_state,dummy,tensorizer,distributed_weight}
--revision REVISION                # 模型版本

# 并行与硬件
--tensor-parallel-size SIZE        # 张量并行数
--pipeline-parallel-size SIZE      # 流水线并行数
--gpu-memory-utilization FRACTION  # GPU 内存利用率 (0-1)
--block-size {8,16,32}             # KV Cache 块大小
--enforce-eager                    # 强制 eager 模式
--max-model-len MAX_MODEL_LEN      # 最大模型长度

# 性能
--max-num-batched-tokens N         # 每批最大 token 数
--max-num-seqs N                   # 每批最大请求数
--max-logprobs N                   # 最大 logprobs 数

# 服务
--host HOST                        # 监听地址（默认 127.0.0.1）
--port PORT                        # 端口号（默认 8000）
--api-key KEY                      # API 密钥
--served-model-name NAME           # 服务模型名
--max-concurrent-requests N        # 最大并发请求数
--request-timeout SECONDS          # 请求超时秒数

# LoRA
--enable-lora                      # 启用 LoRA
--max-loras N                      # 最大 LoRA 数
--max-lora-rank N                  # 最大 LoRA rank
--lora-extra-vocab-path PATH       # LoRA 额外词表路径

# 推测解码
--speculative-model MODEL          # 草稿模型
--num-speculative-tokens N         # 草稿 token 数

# 编译优化
--compile-size SIZE [SIZE ...]     # 编译的 batch size 列表
--disable-log-stats                # 禁用日志统计
--disable-log-requests             # 禁用请求日志

# 多模态
--mm-processor-kwargs JSON         # 多模态处理器参数（JSON 格式）

# 配置
--config PATH                      # YAML 配置文件路径
```

---

## 六、陌生模型部署指南

当遇到 vLLM 尚未支持的模型时，有以下几种部署策略，按推荐顺序排列：

### 6.1 策略一：使用 `trust_remote_code`（最简单）

如果模型在 HuggingFace 上有对应的 `modeling_xxx.py` 远程代码：

```python
from vllm import LLM

llm = LLM(
    model="username/custom-model",
    trust_remote_code=True,   # 信任 HuggingFace 远程代码
    tensor_parallel_size=1,
)
```

> **适用场景：** 模型架构与已有模型接近，HuggingFace 提供了 `modeling_xxx.py`。
> **限制：** 远程代码可能无法与 vLLM 的推理引擎兼容（缺少 vLLM 特定接口）。

### 6.2 策略二：使用 GGUF 格式（无需修改代码）

将模型转换为 GGUF 格式后直接加载：

```python
from vllm import LLM

llm = LLM(
    model="/path/to/model.gguf",
    load_format="gguf",       # 指定 GGUF 加载格式
    dtype="auto",
)
```

转换方式（使用 `llama.cpp`）：

```bash
# 从 HuggingFace 模型转换
python convert-hf-to-gguf.py <model_path> --outfile model.gguf --type q4_k_m
```

> **适用场景：** 量化模型、CPU 推理、资源受限环境。
> **限制：** 需要模型支持 GGUF 格式。

### 6.3 策略三：使用 HuggingFace Config 覆盖

对于需要微调配置的情况：

```python
from vllm import LLM
from vllm.config import HfOverrides

def my_hf_overrides(config):
    """修改 HuggingFace config"""
    config.some_custom_attr = "value"
    return config

llm = LLM(
    model="username/custom-model",
    hf_overrides=my_hf_overrides,   # 回调函数
    # 或 hf_overrides={"some_custom_attr": "value"},  # 字典
)
```

### 6.4 策略四：编写 vLLM 模型适配器（完整支持）

当模型架构独特，需要完整 vLLM 支持时：

#### 步骤 1：实现模型类

参考 [vllm/model_executor/models/llama.py](vllm/model_executor/models/llama.py) 编写模型：

```python
# my_model.py
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import weighted_cache_load

class MyModel(nn.Module):
    """自定义模型实现"""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config

        # 嵌入层
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # 解码层
        self.layers = nn.ModuleList([
            MyDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        ])

        # 输出层
        self.ln = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
        )

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 嵌入
        if input_ids is not None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        # 逐层
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, attn_metadata = layer(x, positions, attn_metadata)

        # 输出
        x = self.ln(x)
        x = self.lm_head(x)
        return x
```

#### 步骤 2：实现解码层

```python
class MyDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.self_attn = Attention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                bias=True,
                prefix=f"{prefix}.mlp.gate_up_proj",
                output_partition_sizes=(
                    config.intermediate_size,
                    config.intermediate_size,
                ),
                vllm_config=vllm_config,
            ),
            ColumnParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                bias=True,
                prefix=f"{prefix}.mlp.down_proj",
                vllm_config=vllm_config,
            ),
        )
        self.norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        # RMSNorm + Attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            attn_metadata=attn_metadata,
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        # RMSNorm + MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

#### 步骤 3：注册模型

**方式 A：插件方式（推荐，不修改 vLLM 源码）**

```python
# my_plugin.py
def register():
    from vllm import ModelRegistry
    from my_model import MyModelForCausalLM

    ModelRegistry.register_model("MyModelForCausalLM", MyModelForCausalLM)
```

在 `pyproject.toml` 中注册 entry point：

```toml
[project.entry-points."vllm.model"]
my_model = "my_plugin:register"
```

**方式 B：直接修改 vLLM 源码**

将模型文件放入 `vllm/model_executor/models/`，在 [registry.py](vllm/model_executor/models/registry.py) 的 `_VLLM_MODELS` 字典中添加：

```python
# vllm/model_executor/models/registry.py
_VLLM_MODELS = {
    # ... existing models ...
    "MyModelForCausalLM": MyModelForCausalLM,
}
```

#### 步骤 4：加载模型

```python
from vllm import LLM

llm = LLM(
    model="/path/to/my/model",
    # 如果 config.json 中 architecture 不匹配，可强制指定：
    # model_type="my_model",  # 通过 hf_overrides 设置
)
```

### 6.5 策略五：使用自定义 Worker 类

对于需要特殊执行逻辑的模型，可以传入自定义 `worker_cls`：

```python
from vllm import LLM

llm = LLM(
    model="base-model",
    worker_cls="my_package.custom_worker.MyWorker",  # 完全自定义 worker
)
```

### 6.6 陌生模型部署决策树

```
遇到陌生模型
    │
    ├── 架构与 Llama/Qwen/Gemma 等相似？
    │       ├── 是 → 尝试修改已有模型文件（复制 + 微调）
    │       └── 否 ↓
    │
    ├── HuggingFace 有 modeling_xxx.py？
    │       ├── 是 → trust_remote_code=True 尝试
    │       └── 否 ↓
    │
    ├── 可以转换为 GGUF？
    │       ├── 是 → load_format="gguf"
    │       └── 否 ↓
    │
    ├── 架构简单（标准 Transformer）？
    │       ├── 是 → 参考 llama.py 编写 vLLM 模型适配器
    │       └── 否 ↓
    │
    └── 架构复杂（Mamba、混合注意力等）？
            ├── 是 → 参考对应已有模型（如 jamba.py、mamba.py）
            └── 仍无法支持 → 提交 Issue + PR 到 vLLM
```

### 6.7 模型实现要点清单

实现新模型时需要注意的关键点：

1. **构造函数签名**：`__init__(self, vllm_config: VllmConfig, prefix: str = "")`
2. **子模块 prefix**：所有 vLLM 内部模块需传入 `prefix=f"{prefix}.module_name"`
3. **forward 签名**：`forward(input_ids, positions, intermediate_tensors, inputs_embeds)`
4. **张量并行**：使用 `VocabParallelEmbedding`、`ParallelLMHead`、`ColumnParallelLinear`、`RowParallelLinear`
5. **权重加载**：实现 `load_weights(state_dict)` 方法处理权重映射
6. **模型注册**：通过 `ModelRegistry.register_model()` 注册
7. **配置映射**：在 `vllm/model_executor/models/config.py` 的 `MODELS_CONFIG_MAP` 中添加运行时默认配置

---

## 七、常见问题

### Q1: OOM（显存溢出）怎么办？

```python
llm = LLM(
    model="large-model",
    gpu_memory_utilization=0.7,    # 降低 GPU 内存利用率
    max_model_len=2048,            # 限制最大长度
    max_num_batched_tokens=1024,   # 限制每批 token 数
)
```

或使用 CPU 卸载：
```python
llm = LLM(
    model="large-model",
    cpu_offload_gb=10,             # 将 10GB 权重卸载到 CPU
)
```

### Q2: 如何加速推理？

```python
llm = LLM(
    model="my-model",
    compilation_config=1,          # 启用 torch.compile
    # 或
    compilation_config={"inductor_compile_size": [1, 2, 4, 8, 16]},
)
```

### Q3: 如何指定注意力后端？

```python
from vllm import LLM

llm = LLM(
    model="my-model",
    attention_config={
        "backend": "flashinfer",     # "flashinfer" | "flash_attn" | "flex_attention" | "triton"
    },
)
```

### Q4: 如何查看已支持的模型？

```bash
# 查看模型列表
python -c "from vllm import ModelRegistry; print(ModelRegistry.get_supported_archs())"

# 或查看文档
# https://docs.vllm.ai/en/latest/models/supported_models.html
```

### Q5: 如何从本地路径加载模型？

```python
llm = LLM(model="/path/to/local/model")

# 指定下载目录
llm = LLM(
    model="huggingface/model-name",
    download_dir="/path/to/cache",
)
```

---

## 八、参考文档

| 文档 | 说明 |
|------|------|
| [EngineArgs](../configuration/engine_args.md) | 引擎参数完整文档 |
| [ServeArgs](../configuration/serve_args.md) | 服务参数文档 |
| [离线推理](../serving/offline_inference.md) | 离线推理指南 |
| [OpenAI 兼容服务器](../serving/openai_compatible_server.md) | API Server 指南 |
| [模型实现教程](../contributing/model/basic.md) | 如何添加新模型 |
| [模型注册](../contributing/model/registration.md) | 模型注册方法 |
| [配置系统](../configuration/README.md) | 配置系统总览 |
| [环境变量](../configuration/env_vars.md) | VLLM_* 环境变量 |
