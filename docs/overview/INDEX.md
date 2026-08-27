# 模块梳理索引

按 [vLLM 系统架构总览](./00_系统架构总览.md) 的模块分类组织。建议先读总览，再按主题深入。

## 总览

- [vLLM 系统架构总览](./00_系统架构总览.md) — 分层架构、进程模型、核心组件、完整请求时序、KV/批处理/编译/并行数据流（**建议先读**）
- [项目结构总览](./README.md) — 顶层目录与 `vllm/` 各子包说明

## 入门与排错（getting-started/）

- [LLM 启动与部署指南](./getting-started/02_llm_startup_guide.md) — 启动方式、参数传递、重要参数、陌生模型部署
- [Qwen 系列排错](./getting-started/03a_qwen_troubleshooting.md)
- [DeepSeek 系列排错](./getting-started/03b_deepseek_troubleshooting.md)
- [GLM/MiniMax 系列排错](./getting-started/03c_glm_minimax_troubleshooting.md)
- [NVIDIA GPU / 昇腾 910B 排错](./getting-started/03d_hardware_troubleshooting.md)

## V1 引擎核心（core/） — `vllm/v1/`

1. [V1 引擎核心](./core/01_engine_core.md) — `vllm/v1/engine/`（AsyncLLM、EngineCore、ZMQ）
2. [调度与 KV Cache](./core/02_scheduler_kv_cache.md) — `vllm/v1/core/`（Scheduler、KVCacheManager、PagedAttention）
3. [注意力后端](./core/03_attention_backends.md) — `vllm/v1/attention/`
4. [Worker 与模型运行器](./core/04_worker_model_runner.md) — `vllm/v1/worker/`（GPUWorker、GPUModelRunner）

## 模型与层（models/） — `vllm/model_executor/`

1. [模型执行与层](./models/05_model_executor_layers.md) — `layers/`（并行 Linear、Attention、MoE、Norm、RoPE、SSM、Pooler）
2. [模型实现](./models/06_model_implementations.md) — `models/`（ModelRegistry、Supports* 协议、290+ 模型）

## 性能与编译（performance/） — `vllm/compilation/`

1. [编译管线](./performance/07_compilation.md) — torch.compile 分段后端、Inductor pass、CUDA Graph

## 分布式（distributed/） — `vllm/distributed/`

1. [分布式推理](./distributed/08_distributed.md) — TP/PP/EP/DP/CP、通信组、KV/EC connector、EPLB、权重传输

## 服务化（serving/） — `vllm/entrypoints/`、`vllm/v1/engine/` AsyncLLM

1. [之一：启动与架构总览](./serving/api-server-01-启动与架构.md)
2. [之二：请求处理流水线](./serving/api-server-02-请求流水线.md)
3. [之三：OpenAI 接口实现](./serving/api-server-03-openai-接口.md)
4. [之四：批处理与扩展服务](./serving/api-server-04-批处理与扩展.md)
5. [之五：配置与部署](./serving/api-server-05-配置与部署.md)

## 可观测性（observability/）

- [日志系统详解](./observability/04_logging_system.md) — 日志格式、输出目录、KV Cache 监控、Prometheus/OTel 指标
- Tracing 系列（`vllm/tracing/`）：
    - [之一：架构与公共 API](./observability/tracing/tracing-01-架构与公共API.md)
    - [之二：OpenTelemetry 后端实现](./observability/tracing/tracing-02-otel后端实现.md)
    - [之三：上下文传播与埋点实践](./observability/tracing/tracing-03-上下文传播与埋点实践.md)

## 本地实验环境

- [lab/](../../lab/) — Docker + CPU 模式本地实验环境（离线推理 / API Server / 参数对比 / 日志探索 / curl 测试）

## 待梳理模块

1. LoRA 支持 — `vllm/lora/`
2. 多模态 — `vllm/multimodal/`
3. 量化 — `vllm/model_executor/layers/quantization/`
4. 配置系统 — `vllm/config/`
5. 原生算子 — `csrc/`
6. 分词器与解析器 — `vllm/tokenizers/`、`vllm/tool_parsers/`
