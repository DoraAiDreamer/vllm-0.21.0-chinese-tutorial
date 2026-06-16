# 源码解读索引

按模块逐步深入，每完成一个模块在此更新。

## 总览

- [项目结构总览](./README.md)

## 使用指南

1. [LLM 启动与部署指南](./docs/overview/02_llm_startup_guide.md) — 启动方式、参数传递、重要参数含义、陌生模型部署
2. [Qwen 系列排错](./docs/overview/03a_qwen_troubleshooting.md) — Qwen/Qwen2/Qwen2.5/Qwen3 全系列部署常见问题与解决方案
3. [DeepSeek 系列排错](./docs/overview/03b_deepseek_troubleshooting.md) — DeepSeek V2/V3/V4 全系列部署常见问题与解决方案
4. [GLM/MiniMax 系列排错](./docs/overview/03c_glm_minimax_troubleshooting.md) — GLM-4/ChatGLM/MiniMax 全系列部署常见问题与解决方案
5. [NVIDIA GPU / 昇腾 910B 排错](./docs/overview/03d_hardware_troubleshooting.md) — 硬件部署通用问题、显存管理、通信调试、性能优化
6. [日志系统详解](./docs/overview/04_logging_system.md) — 日志格式、自定义输出目录、KV Cache 用量监控、Prometheus/OpenTelemetry 指标

## 待梳理模块

1. [V1 引擎核心](./docs/overview/01_engine_core.md) — `vllm/v1/engine/`
2. [调度与 KV Cache](./docs/overview/02_scheduler_kv_cache.md) — `vllm/v1/core/`
3. [注意力后端](./docs/overview/03_attention_backends.md) — `vllm/v1/attention/`
4. [Worker 与模型运行器](./docs/overview/04_worker_model_runner.md) — `vllm/v1/worker/`
5. [模型执行与层](./docs/overview/05_model_executor_layers.md) — `vllm/model_executor/layers/`
6. [模型实现](./docs/overview/06_model_implementations.md) — `vllm/model_executor/models/`
7. [编译管线](./docs/overview/07_compilation.md) — `vllm/compilation/`
8. [分布式推理](./docs/overview/08_distributed.md) — `vllm/distributed/`
9. [LoRA 支持](./docs/overview/09_lora.md) — `vllm/lora/`
10. [多模态](./docs/overview/10_multimodal.md) — `vllm/multimodal/`
11. [量化](./docs/overview/11_quantization.md) — `vllm/model_executor/layers/quantization/`
12. [入口 API](./docs/overview/12_entrypoints.md) — `vllm/entrypoints/`
13. [配置系统](./docs/overview/13_config.md) — `vllm/config/`
14. [原生算子](./docs/overview/14_native_kernels.md) — `csrc/`
15. [分词器与解析器](./docs/overview/15_tokenizers_parsers.md) — `vllm/tokenizers/`、`vllm/tool_parsers/`
