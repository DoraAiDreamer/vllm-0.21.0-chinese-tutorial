# API Server 之四：批处理与扩展服务

> 相关源码：`vllm/entrypoints/openai/run_batch.py`、`vllm/entrypoints/serve/`、`vllm/entrypoints/cli/run_batch.py`

本篇覆盖两类内容：
1. **离线批处理**：OpenAI 兼容的 File/Batch API（`vllm run-batch`）；
2. **管理与扩展 HTTP API**：`vllm/entrypoints/serve/` 下挂载的非 OpenAI 标准端点——PD 分离、弹性 EP、LoRA、sleep/wake、RLHF 权重更新、profile、cache、collective RPC、tokenize、render、instrumentator。

---

## 1. OpenAI 批处理（run_batch）

OpenAI 的 Batch API 允许上传一个 JSONL 文件，每行是一个 `/v1/chat/completions` 或 `/v1/completions` 请求，异步处理后下载结果。vLLM 用 `run_batch.py` 实现这一流程。

### 1.1 CLI 入口

- `vllm/entrypoints/cli/run_batch.py` 注册 `vllm run-batch` 子命令；
- `run_batch.py:make_arg_parser`/`parse_args` 定义参数：输入文件/URL、输出 URL、模型、endpoint、并发、采样参数等；
- `BatchFrontendArgs`（[L229](../../vllm/entrypoints/openai/run_batch.py#L229)）封装参数并自定义 CLI 选项。

### 1.2 数据模型

| 类 | 作用 |
|----|------|
| `BatchRequestInput`（[L145](../../vllm/entrypoints/openai/run_batch.py#L145)） | JSONL 里的一行：`custom_id`、`method`、`url`、`body` |
| `BatchResponseData`（[L199](../../vllm/entrypoints/openai/run_batch.py#L199)） | 响应体（status_code、response 内容） |
| `BatchRequestOutput`（[L210](../../vllm/entrypoints/openai/run_batch.py#L210)） | 最终 JSONL 输出行：`custom_id`、`response`、`error` |
| `BatchTranscriptionRequest`/`BatchTranslationRequest` | 音频转写/翻译批请求的校验变体（要求无内嵌文件） |
| `BatchProgressTracker`（[L304](../../vllm/entrypoints/openai/run_batch.py#L304)） | 用 tqdm 显示 submitted/completed 进度 |

### 1.3 执行流程

1. **读输入**：`read_file(path_or_url)` 支持本地路径与 http(s) URL；
2. **建引擎**：内部构造 `AsyncLLM`/`EngineClient`；
3. **端点注册表**：`build_endpoint_registry`（[L693](../../vllm/entrypoints/openai/run_batch.py#L693)）把 URL（`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、转录/翻译等）映射到处理函数；`handle_endpoint_request`（[L582](../../vllm/entrypoints/openai/run_batch.py#L582)）按 method/url 分发；
4. **并发执行**：`run_request`（[L535](../../vllm/entrypoints/openai/run_batch.py#L535)）对每一行调对应 serving 方法，用 `asyncio` 并发；
5. **错误处理**：`make_error_request_output`/`make_async_error_request_output` 把异常转成 Batch 错误格式（不中断整批）；
6. **写输出**：`write_file`/`upload_data` 支持本地文件、云存储 URL（`output_url`）。

`run_batch` 复用在线 serving 类（`OpenAIServingChat`/`Completion`/`Responses`），所以输出格式与在线 API 完全一致，只是没有流式 SSE。

### 1.4 在线 Batch 端点

除了 CLI，`chat_completion/batch_serving.py` 提供 `POST` 批量 chat 端点（`api_router` 里的 `create_batch_chat_completion`，[api_router.py:77-103](../../vllm/entrypoints/openai/chat_completion/api_router.py#L77)），一次请求里带多个 prompt 并行处理。

---

## 2. vLLM Serve 管理 API

[serve/__init__.py:12](../../vllm/entrypoints/serve/__init__.py#L12) 的 `register_vllm_serve_api_routers` 在 `build_app` 阶段无条件挂载一批管理端点（不同于 OpenAI 端点按 task 挂载）。

### 2.1 LoRA 管理（`serve/lora/`）

`api_router.py` 提供：
- `POST /v1/load_lora_adapter`：运行时加载 LoRA（请求体含 `lora_name`、`lora_path` 等），调 `engine_client.add_lora`；
- `POST /v1/unload_lora_adapter`：卸载，调 `remove_lora`；
- 还有 `pin_lora_adapter` 等（见 [lora/api_router.py:58](../../vllm/entrypoints/serve/lora/api_router.py#L58)）。

这些是 OpenAI `/v1/models` 之外更底层的 LoRA 控制端点，`protocol.py` 定义请求体。启动时 `--lora-modules` 加载的静态 LoRA 由 `OpenAIServingModels.init_static_loras` 处理（第 3 篇）。

### 2.2 Sleep / Wake（`serve/sleep/`）

- `POST /sleep`：让引擎释放权重和/或 KV cache 显存（level 1/2、mode abort/wait/keep），调 `engine_client.sleep`；
- `POST /wake_up`：恢复，调 `wake_up`；
- `GET /is_sleeping`：查询状态。

用于多模型分时复用 GPU、与 `vllm serve --sleep` 配合。底层是第 04 篇 Worker 的 CuMemAllocator 机制。

### 2.3 性能分析（`serve/profile/`）

- `POST /start_profile`：开始 torch profiler/trace；
- `POST /stop_profile`：停止并落盘。

透传到 `engine_client.start_profile/stop_profile`。

### 2.4 Cache 管理（`serve/cache/`）

- `POST /reset_prefix_cache`：清空前缀缓存；
- `POST /reset_mm_cache`：清空多模态编码器缓存；
- `POST /reset_encoder_cache`：清空 encoder cache。

### 2.5 Collective RPC（`serve/rpc/`）

`POST /collective_rpc`：在所有 worker 上调用任意方法（body 含 `method`、`args`、`kwargs`、`group`），调 `engine_client.collective_rpc`。用于调试、运维、动态配置。

### 2.6 Tokenize / Detokenize（`serve/tokenize/`）

- `POST /tokenize`：把文本/prompt 转成 token id；
- `POST /detokenize`：把 token id 转回文本；
- `GET /tokenizer_info`：返回 tokenizer 元信息（eos、词表大小等）。

由 `OpenAIServingTokenization` 实现，直接用渲染器的 tokenizer，不经过 GPU。

### 2.7 Render（`serve/render/`）

`POST` 渲染端点：只应用 chat template、工具解析、token 计数，不做推理。用于调试模板与估算 token。第 3 篇提到的"render"任务即此。

### 2.8 Instrumentator / 指标（`serve/instrumentator/`）

`register_instrumentator_api_routers` 挂载 Prometheus/指标相关端点与中间件，配合 `orca_metrics.py`。

---

## 3. PD 分离（disagg）

[serve/disagg/](../../vllm/entrypoints/serve/disagg/) 提供 Prefill-Decode 分离的**生成端点**（区别于 KV connector 这种传输层）：

- `POST /inference/v1/generate`（[api_router.py:49](../../vllm/entrypoints/serve/disagg/api_router.py#L49)）：返回 `text/event-stream`；
- `POST /abort_requests`：中止请求；
- `protocol.py`：`GenerateRequest`/`GenerateResponse`；
- `serving.py`：`ServingTokens`，协调 prefill 实例与 decode 实例之间的 KV 传输（通过 KV connector）和流式输出。

这是 vLLM 自带的一个简单 PD 路由/生成接口；生产中也常用外部代理（如路由器）配合 KV connector 实现更复杂的调度。

---

## 4. 弹性专家并行（elastic_ep）

[serve/elastic_ep/api_router.py](../../vllm/entrypoints/serve/elastic_ep/api_router.py)：
- `POST /scale_elastic_ep`：触发 EP 扩缩容，传新的 data parallel size；调 `engine_client.scale_elastic_ep`（第 08 篇的 Elastic EP）；
- `POST /is_scaling_elastic_ep`：查询是否正在扩缩容。

这让上层编排器（K8s/控制器）可以通过 HTTP 动态增减 MoE 专家并行的 worker。

---

## 5. RLHF / 在线权重更新（rlhf）

[serve/rlhf/api_router.py](../../vllm/entrypoints/serve/rlhf/api_router.py) 暴露第 04/08 篇的权重传输与生成暂停能力：

- `POST /pause`：暂停生成（`pause_generation`）；
- `POST /resume`：恢复（`resume_generation`）；
- `GET /is_paused`；
- `POST /init_weight_transfer_engine`：建立与训练端的连接；
- `POST /start_weight_update`：开始一次权重更新；
- `POST /update_weights`：推送一批权重；
- `POST /finish_weight_update`：完成并加载；
- `GET /get_world_size`：查询分布式 world size。

典型用于 RLHF/在线训练：训练端周期性把新权重推给推理引擎，期间暂停调度，更新完恢复。

---

## 6. 其他入口与适配

- **`entrypoints/sagemaker/`**：SageMaker 兼容路由（`/invocations`、`/ping`），由 `build_app` 通过 `register_sagemaker_api_router` 挂载；
- **`entrypoints/anthropic/`**：Anthropic Messages API 兼容层；
- **`entrypoints/grpc_server.py`**：gRPC 入口（可选）；
- **`entrypoints/mcp/`**：Model Context Protocol 端点；
- **`entrypoints/pooling/`**：池化模型专用路由工厂（embedding/score/classify、late-interaction），按任务挂载；
- **`entrypoints/llm.py`**：离线 `LLM` 类的入口（非 HTTP）。

---

## 小结

1. **`run_batch` 复用在线 serving 类**实现 OpenAI 兼容的离线 JSONL 批处理，支持本地/云端输入输出、并发、错误隔离；
2. **`serve/` 无条件挂载一组管理 API**：LoRA 热加载、sleep/wake、profile、cache reset、collective RPC、tokenize、render、指标；
3. **disagg 端点**提供简单的 PD 分离生成接口；
4. **elastic_ep 端点**支持 MoE 专家并行的 HTTP 扩缩容；
5. **rlhf 端点**暴露暂停/恢复与在线权重传输，服务训练-推理联动；
6. 还有 SageMaker、Anthropic、gRPC、MCP、pooling 等适配入口。

下一篇 [API Server 之五：配置与部署](./api-server-05-配置与部署.md) 将汇总 CLI 参数、SSL/鉴权中间件、日志与指标、Uvicorn 部署选项，以及与 EngineCore/分布式的协作拓扑。
