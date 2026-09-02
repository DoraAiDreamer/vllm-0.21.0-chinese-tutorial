# vLLM 项目结构总览

> 基于 vllm-0.21.0 版本

## 一、项目简介

vLLM 是一个开源的大语言模型（LLM）推理和服务框架，核心特性包括：

- **高吞吐推理**：通过 PagedAttention 算法实现高效的 KV Cache 内存管理
- **连续批处理**：支持 chunked prefill、prefix caching、动态请求调度
- **多硬件支持**：NVIDIA GPU（CUDA）、AMD GPU（ROCm）、Intel GPU（XPU）、TPU、CPU
- **200+ 模型架构**：覆盖 Decoder-only、MoE、混合注意力/状态空间、多模态、Embedding、Reward 等
- **多种量化方案**：FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ、AWQ、GGUF 等
- **丰富的 API**：OpenAI 兼容 API、Anthropic Messages API、gRPC、CLI

## 二、顶层目录结构

```
vllm-0.21.0/
├── vllm/                  # 核心 Python 包（推理引擎）
├── csrc/                  # C++/CUDA 原生代码（自定义算子）
├── tests/                 # 测试套件
├── examples/              # 使用示例
├── benchmarks/            # 性能基准测试
├── docker/                # Docker 镜像构建
├── scripts/               # 构建/开发脚本
├── cmake/                 # CMake 构建配置
├── requirements/          # 依赖规格文件
├── tools/                 # 工具（Helion 内核自动调优）
├── docs/                  # 文档（本目录）
├── CMakeLists.txt         # CMake 主构建文件
├── setup.py               # Python 包安装脚本
├── pyproject.toml         # 项目元数据与工具配置
└── AGENTS.md              # AI 助手贡献规范
```

## 三、核心 Python 包 (`vllm/`)

### 3.1 顶层文件 — 公共入口

| 文件 | 说明 |
|------|------|
| `__init__.py` | 公共 API 导出（LLM、ModelRegistry、SamplingParams、EngineArgs 等），使用懒加载机制 |
| `version.py` | 版本号 |
| `env_override.py` / `envs.py` | 环境变量管理（100+ 个 VLLM_* 变量） |
| `sampling_params.py` | 采样参数（temperature、top_p、top_k、结构化输出等） |
| `outputs.py` | 输出数据结构（CompletionOutput、PoolingOutput 等） |
| `inputs/` | 输入处理（TextPrompt、TokensPrompt、多模态输入） |
| `sequence.py` | 流水线并行中间状态 |
| `beam_search.py` | 束搜索算法实现 |
| `forward_context.py` | 前向传播全局上下文管理器 |

### 3.2 `vllm/v1/` — V1 执行引擎（默认引擎）

这是 v0.21.0 的**默认且活跃维护**的引擎版本。顶层 `vllm.engine` 和 `vllm.entrypoints.llm` 均指向 v1 实现。

```
vllm/v1/
├── engine/           # 引擎核心
│   ├── llm_engine.py     # LLMEngine — 引擎主循环
│   ├── core.py           # EngineCore — 实际推理循环
│   ├── core_client.py    # IPC 通信客户端
│   ├── input_processor.py  # 输入处理（分词、多模态预处理）
│   ├── output_processor.py # 输出后处理、去分词
│   ├── detokenizer.py    # 去分词逻辑
│   ├── coordinator.py    # 协调器线程
│   └── parallel_sampling.py # 并行采样
├── executor/         # 执行器后端
│   ├── uniproc_executor.py   # 单进程执行器
│   ├── ray_executor.py       # Ray 分布式执行器
│   ├── ray_executor_v2.py    # V2 Ray 执行器
│   └── multiproc_executor.py # 多进程执行器
├── worker/           # Worker 实现
│   ├── gpu_worker.py         # GPU Worker
│   ├── gpu_model_runner.py   # GPU 模型运行器（加载模型、执行前向传播）
│   ├── cpu_worker.py         # CPU Worker
│   ├── xpu_worker.py         # XPU Worker
│   ├── cudagraph_dispatcher.py # CUDA 图管理
│   └── gpu/                  # GPU 特定组件
│       ├── model_states/     # 模型状态管理
│       ├── spec_decode/      # 推测解码
│       └── mm/               # 多模态编码器处理
├── core/             # 核心调度与 KV Cache 管理
│   ├── sched/              # 调度器（请求调度、批处理、KV Cache 分配）
│   │   ├── scheduler.py
│   │   ├── request_queue.py
│   │   └── output.py
│   ├── kv_cache_manager.py # KV Cache 分配与管理
│   ├── kv_cache_coordinator.py
│   └── block_pool.py       # KV Cache 块池管理
├── attention/        # 注意力后端
│   ├── backend.py          # AttentionBackend 基类
│   ├── backends/           # 具体实现
│   │   ├── flashinfer.py   # FlashInfer 后端
│   │   ├── flash_attn.py   # FlashAttention 后端
│   │   ├── flex_attention.py # PyTorch Flex Attention
│   │   ├── mamba_attn.py   # Mamba 注意力
│   │   ├── cpu_attn.py     # CPU 注意力
│   │   ├── mla/            # MLA（多隐注意力）实现
│   │   ├── registry.py     # 后端注册表
│   │   └── selector.py     # 后端自动选择
│   └── cache/              # 注意力缓存管理
├── sample/           # 采样模块
│   ├── sampler.py          # Token 采样器
│   ├── logits_processor/   # Logit 处理
│   └── ops/                # 采样算子
├── structured_outputs/  # 结构化/引导解码
├── speculative.py     # 推测解码编排
├── lora.py            # LoRA 集成
├── multimodal.py      # 多模态支持
├── mamba.py           # Mamba 状态空间模型
├── speech_to_text.py  # 语音识别
├── reasoning.py       # 推理/思维链
├── quantization.py    # 量化支持
├── pooler.py          # Pooler
├── observability.py   # 可观测性（追踪/指标）
├── offload.py         # CPU 卸载
├── parallel.py        # 并行执行
├── compilation.py     # 编译设置
├── kv_transfer.py     # KV Cache 传输
├── ec_transfer.py     # 弹性容器传输
├── weight_transfer.py # 权重传输
├── kernel.py          # 内核抽象
├── device.py          # 设备抽象
├── load.py            # 模型加载
├── model.py           # 模型抽象
└── model_arch.py      # 模型架构定义
```

### 3.3 `vllm/model_executor/` — 模型执行基础设施

#### 3.3.1 `layers/` — 神经网络层

```
vllm/model_executor/layers/
├── attention/              # 注意力层
│   ├── attention.py            # 主注意力层
│   ├── mla_attention.py        # 多隐注意力（MLA）
│   ├── cross_attention.py      # 交叉注意力
│   └── encoder_only_attention.py # 编码器专用注意力
├── fused_moe/              # 融合 MoE 层（MoE 执行核心）
│   ├── layer.py              # FusedMoE 层
│   ├── fused_moe.py          # 核心融合 MoE 实现
│   ├── triton_deep_gemm_moe.py # DeepGEMM 版
│   ├── triton_cutlass_moe.py   # Cutlass 版
│   ├── cpu_fused_moe.py      # CPU 回退
│   ├── experts/              # 专家实现
│   ├── router/               # 路由器实现
│   ├── runner/               # MoE 运行器
│   └── oracle/               # 内核选择优化器
├── layernorm.py            # 层归一化
├── linear.py               # 并行线性层（列/行/复制）
├── vocab_parallel_embedding.py # 词表嵌入
├── rotary_embedding/       # 旋转位置编码（20+ 实现）
├── mamba/                  # Mamba 状态空间层
├── quantization/           # 量化层
├── pooler/                 # Pooling 层
└── deepseek_v4_attention.py    # DeepSeek V4 注意力
```

#### 3.3.2 `layers/quantization/` — 量化方法（35+ 种）

FP8、GPTQ、GPTQ-Marlin、AWQ、AWQ-Marlin、BitsAndBytes、GGUF、Marlin、TorchAO、ModelOpt、MXFP4、Quark、TurboQuant、在线量化等。

#### 3.3.3 `models/` — 模型实现（290+ 个）

```
vllm/model_executor/models/
├── registry.py         # 模型注册表（映射 HuggingFace AutoModel 到 vLLM 实现）
├── config.py           # 模型配置覆盖
├── interfaces.py       # 模型接口协议（HasInnerState, SupportsLoRA 等）
├── interfaces_base.py  # 基础接口（VllmModelForTextGeneration, VllmModelForPooling）
└── [290+ 模型文件]
```

**模型分类：**

| 类别 | 代表模型 |
|------|----------|
| Decoder-only LLM | Llama、Qwen、Gemma、GLM、DeepSeek、Phi、Cohere、Falcon 等 |
| MoE LLM | Mixtral、DeepSeek-V3/V4、Qwen-MoE、GPT-OSS 等 |
| 视觉多模态 | LLaVA、Qwen-VL、Pixtral、Gemma3、InternVL 等 |
| 音频/语音 | Whisper、Qwen-Audio、Kimi-Audio 等 |
| 多模态全栈 | Gemma3n、Qwen2.5/3-Omni 等 |
| Embedding/Reranker | BERT、ColBERT、GTE、BGE、Jina 等 |
| Reward/Classification | Qwen-Math 等 |
| OCR | ColQwen、GLM-OCR、PaddleOCR-VL 等 |

#### 3.3.4 `model_loader/` — 模型加载

模型权重加载策略（safetensors、远程加载、分片加载、自定义全归约等）。

### 3.4 `vllm/entrypoints/` — 高层入口 API

```
vllm/entrypoints/
├── llm.py              # LLM 类 — 用户主入口 API
├── api_server.py       # OpenAI 兼容 API 服务器
├── openai/             # OpenAI API 兼容层
│   ├── chat_completion/    # 聊天补全端点
│   ├── completion/         # 文本补全端点
│   ├── responses/          # Responses API
│   ├── speech_to_text/     # 语音转文本端点
│   ├── realtime/           # 实时 API
│   ├── models/             # 模型列表端点
│   ├── engine/             # 引擎客户端管理
│   └── parser/             # 请求解析
├── cli/              # 命令行接口
│   ├── main.py           # CLI 主入口
│   ├── openai.py         # OpenAI 服务器 CLI
│   └── anthropic/        # Anthropic API 服务器
├── serve/            # vLLM 服务框架
│   ├── middleware/       # 请求/响应中间件
│   ├── plugins/          # 插件系统
│   ├── profile/          # 性能分析中间件
│   ├── sleep/            # 请求休眠/限流
│   ├── instrumentator/   # Prometheus 指标
│   └── disagg/           # 分离式服务（预填充/解码分离）
├── grpc_server.py    # gRPC 服务器
├── anthropic/        # Anthropic Messages API
├── mcp/              # Model Context Protocol 支持
├── chat_utils.py     # 聊天消息解析
├── launcher.py       # 分布式启动器
└── pooling/          # Pooling 模型入口
```

### 3.5 `vllm/config/` — 配置系统

VllmConfig 是一个组合数据类，聚合了所有子配置：

| 配置模块 | 内容 |
|----------|------|
| `vllm.py` | 主配置类（聚合所有子配置） |
| `model.py` | 模型路径、类型、分词器、最大长度 |
| `cache.py` | KV Cache 类型、块大小、GPU 内存利用率 |
| `attention.py` | 注意力后端选择 |
| `scheduler.py` | 调度参数、最大批处理 token 数 |
| `load.py` | 加载架构、分发策略 |
| `parallel.py` | 张量/流水线/数据/专家并行 |
| `compilation.py` | torch.compile 模式、CUDA 图设置 |
| `lora.py` | LoRA 配置 |
| `speculative.py` | 推测解码配置 |
| `multimodal.py` | 多模态设置 |
| `offload.py` | CPU 卸载 |
| `kv_transfer.py` | KV Cache 传输 |
| `quantization/` | 运行时量化配置解析 |

### 3.6 `vllm/compilation/` — 编译管线

```
vllm/compilation/
├── compiler_interface.py   # 编译接口
├── piecewise_backend.py    # 分段编译后端
├── cuda_graph.py         # CUDA 图管理
├── wrapper.py            # 编译封装
└── passes/               # 编译 Pass
    ├── fusion/           # 融合 Pass（激活量化、AllReduce、RoPE 等）
    ├── ir/               # IR Pass
    └── utility/          # 工具 Pass
```

### 3.7 `vllm/distributed/` — 分布式推理

张量并行、流水线并行、数据并行、专家并行、KV Cache 传输、权重传输、弹性专家并行等。

### 3.8 `vllm/lora/` — LoRA 支持

```
vllm/lora/
├── model_manager.py      # LoRAModelManager（管理加载的 LoRA 适配器）
├── lora_model.py         # LoRAModel（LoRA 模型封装）
├── lora_weights.py       # LoRALayerWeights
├── worker_manager.py     # LoRA Worker 管理
├── layers/               # LoRA 兼容层封装
├── ops/                  # LoRA 操作
└── punica_wrapper/       # Punica 封装（核心 LoRA 内核分发）
    ├── punica_gpu.py     # GPU Punica（基于 CuPy）
    ├── punica_cpu.py     # CPU Punica
    └── punica_xpu.py     # XPU Punica
```

### 3.9 `vllm/multimodal/` — 多模态支持

多模态注册表、媒体类型处理（图像/视频/音频）、媒体连接器、各模型的处理实现（30+ 处理器）。

### 3.10 `vllm/tokenizers/` — 分词器支持

HuggingFace 分词器封装、模型专用分词器（DeepSeek、Grok2、Mistral 等）、Jinja 聊天模板（15+ 模板）、分词器处理器。

### 3.11 `vllm/tool_parsers/` — 工具调用解析器

40+ 模型专用的工具调用解析器。

### 3.12 `vllm/reasoning/` — 推理/思维链解析器

25+ 模型专用的推理解析器。

### 3.13 `vllm/platforms/` — 平台抽象

Platform 基类，支持 CUDA/ROCm/TPU/XPU/CPU 平台自动检测与分发。

### 3.14 `vllm/utils/` — 工具函数库

异步工具、缓存、计数器、内存管理、NCCL、网络、 profiling、注册表、序列化等 30+ 模块。

### 3.15 `vllm/plugins/` — 插件系统

通过 entry points 加载插件（LoRA 文件系统解析器、HF Hub 解析器等）。

## 四、原生代码 (`csrc/`)

```
csrc/
├── torch_bindings.cpp        # PyTorch C++ 扩展绑定（向 Python 暴露所有 C++ 算子）
├── ops.h                     # C++ 算子声明
├── cache.h                   # KV Cache 数据结构
├── attention/                # 注意力 CUDA 算子
│   ├── paged_attention_v1.cu
│   ├── paged_attention_v2.cu
│   └── mla/                  # MLA 注意力算子
├── cache_kernels*.cu         # KV Cache 内核操作
├── moe/                      # MoE 内核（对齐求和、置换/逆置换、Top-K）
├── quantization/             # 量化内核（AWQ、GGUF、GPTQ、Marlin）
├── mamba/                    # Mamba SSM 内核
├── layernorm_kernels.cu      # 层归一化内核
├── pos_encoding_kernels.cu   # 位置编码内核
├── sampler.cu                # GPU 采样内核
├── custom_all_reduce.cu      # 自定义全归约
├── cumem_allocator.cpp       # cuMEM 分配器
├── cpu/                      # CPU 内核（AMX/NEON/VSX 等架构优化）
│   ├── cpu_attn_*.hpp
│   ├── cpu_fused_moe.cpp
│   └── dnnl_kernels.cpp
├── cutlass_extensions/       # 自定义 Cutlass 工具
├── rocm/                     # ROCm 特定代码
└── libtorch_stable/          # LibTorch 稳定内核
```

## 五、测试体系 (`tests/`)

测试结构镜像 `vllm/` 包，并附加测试基础设施：

| 目录 | 覆盖范围 |
|------|----------|
| `v1/` | V1 引擎测试（注意力、调度器、Worker、KV Cache、推测解码等） |
| `engine/` | 引擎测试 |
| `entrypoints/` | 入口测试（OpenAI、CLI、Serve、gRPC 等） |
| `model_executor/` | 模型执行测试（层、模型加载、量化） |
| `models/` | 模型集成测试 |
| `kernels/` | 算子正确性测试（注意力、Cache、MoE、RoPE 等） |
| `distributed/` | 分布式测试（全归约、NCCL、上下文并行、流水线并行等） |
| `multimodal/` | 多模态测试 |
| `lora/` | LoRA 测试 |
| `quantization/` | 量化测试 |
| `tokenizers/` | 分词器测试 |

## 六、关键架构要点

1. **V1 引擎是默认引擎**：`vllm.engine` 顶层模块是 `vllm.v1` 的薄封装别名
2. **用户入口**：`from vllm import LLM` — 创建 LLM 实例，内部封装 V1 引擎
3. **执行流程**：LLM → EngineCore → Scheduler → ModelRunner → Attention Backend → CUDA Kernels (csrc/)
4. **模型注册**：290+ 模型通过 ModelRegistry 注册，映射 HuggingFace AutoModel 类名到 vLLM 实现
5. **可插拔注意力后端**：FlashInfer、FlashAttention、FlexAttention、ROCm Aiter、Triton、CPU，自动选择或通过配置指定
6. **平台抽象**：`current_platform` 懒检测 CUDA/ROCm/TPU/XPU/CPU，分发所有硬件特定代码

## 七、模块梳理导航

> 根目录 [README.md](../../README.md) 是项目入口；根目录 [INDEX.md](../../INDEX.md) 为跳转页，指向本索引。

详见 [源码解读索引](./INDEX.md)。
