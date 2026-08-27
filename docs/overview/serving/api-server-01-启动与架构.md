# API Server 之一：启动与架构总览

> 相关源码：`vllm/entrypoints/openai/api_server.py`、`vllm/entrypoints/launcher.py`、`vllm/v1/engine/async_llm.py`、`vllm/v1/engine/core_client.py`、`vllm/engine/protocol.py`

## 0. 在整个 vLLM 栈中的位置

答案是：**API Server 在 `vllm/v1/engine/` 之前**。请求流向是：

```
HTTP/WebSocket 客户端
        │  (OpenAI 兼容协议)
        ▼
┌──────────────────────────────────────────────┐
│ API Server 进程（FastAPI + Uvicorn）          │  vllm/entrypoints/openai/
│  · 路由 / 鉴权 / CORS / 异常处理              │
│  · 请求解析、chat template、tool/grammar      │
│  · OpenAIServing* 业务层                      │
└──────────────────────────────────────────────┘
        │  AsyncLLM.generate/encode (async)
        ▼
┌──────────────────────────────────────────────┐
│ AsyncLLM（vllm/v1/engine/async_llm.py）      │  API 进程内
│  · InputProcessor：EngineInput→CoreRequest   │
│  · OutputProcessor：CoreOutput→RequestOutput │
│  · 后台 output_handler 任务                   │
└──────────────────────────────────────────────┘
        │  ZMQ（EngineCoreClient.make_async_mp_client）
        ▼
┌──────────────────────────────────────────────┐
│ EngineCoreProc / DPEngineCoreProc（独立进程） │  vllm/v1/engine/core.py
│  · Scheduler + ModelExecutor + Worker         │
│  · 忙等循环 _process_input_queue / _step      │
└──────────────────────────────────────────────┘
```

关键点：**API Server 和 EngineCore 通常是两个进程**。API 进程跑 FastAPI 事件循环与 AsyncLLM；EngineCore 进程跑调度与 GPU 计算。两者通过 ZMQ 传 `EngineCoreRequest`/`EngineCoreOutputs`（见 `vllm/v1/engine/__init__.py` 的消息定义）。这样 HTTP 慢客户端、tokenization、模板渲染等都不会阻塞 GPU 主循环。

> 例外：`InprocClient`（[core_client.py:274](../../vllm/v1/engine/core_client.py#L274)）用于同进程场景（如测试、`LLM` 类），不经过 ZMQ。

---

## 1. 启动入口

### 1.1 CLI

启动命令 `vllm serve <model> ...` 对应：

- `vllm/entrypoints/cli/serve.py` 注册 `serve` 子命令；
- `vllm/entrypoints/cli/main.py` 解析；
- 最终调用 `vllm/entrypoints/openai/api_server.py:run_server`（[api_server.py:686](../../vllm/entrypoints/openai/api_server.py#L686)）。

`run_server` 主要做三件事：
1. `validate_api_server_args(args)` 校验参数；
2. `setup_server(args)`（[L549](../../vllm/entrypoints/openai/api_server.py#L549)）：建立 `AsyncLLM` 或连接远端 engine（`build_async_engine_client`），`build_app(args, ...)` 构建 FastAPI app，`init_app_state(...)` 注入 serving 状态；
3. `run_server_worker(...)`：调 `launcher.serve_http(...)` 启动 Uvicorn。

### 1.2 build_async_engine_client

[api_server.py:78-157](../../vllm/entrypoints/openai/api_server.py#L78) 的 `build_async_engine_client` / `build_async_engine_client_from_engine_args` 是一个 `@asynccontextmanager`：

- 如果传入了 `--api-server-location` 等远端参数，则创建 `AsyncLLMRaiseForNotImplemented`/远程客户端连接已存在的 engine；
- 否则在本地 `AsyncLLM.from_engine_args(...)`，内部会 `make_async_mp_client` 拉起 EngineCore 子进程。

它同时返回 `supported_tasks`（generate/render/transcription/realtime/pooling 等）和 `model_config`，供 `build_app` 决定挂载哪些路由。

---

## 2. FastAPI 应用装配：build_app

[build_app](../../vllm/entrypoints/openai/api_server.py#L157) 是整个 HTTP 层的组装中心。它创建 `FastAPI(lifespan=lifespan)`，然后按任务类型**挂载路由**、**注册中间件**、**异常处理器**。

### 2.1 路由挂载顺序

```python
register_vllm_serve_api_routers(app)              # vllm/entrypoints/serve 通用管理 API
register_models_api_router(app)                  # GET /v1/models
register_sagemaker_api_router(app, tasks, cfg)   # SageMaker 兼容路由
if "generate" in tasks:
    register_generate_api_routers(app)           # /v1/completions, /v1/chat/completions
    attach_disagg_router(app)                    # PD 分离
    attach_rlhf_router(app)                      # RLHF
    elastic_ep_attach_router(app)                # 弹性 EP
    register_generative_scoring_api_router(app)  # 生成式评分
if "generate" or "render":
    attach_render_router(app)                    # 模板渲染（无推理）
if "transcription": register_speech_to_text_api_router(app)
if "realtime":      register_realtime_api_router(app)   # WebSocket 实时语音
if any pooling task: register_pooling_api_routers(...)  # /pooling, /score, embed 等
```

每个子模块一般有三个文件：
- `protocol.py`：Pydantic 请求/响应模型；
- `serving.py`：`OpenAIServing*` 业务类，调 `engine_client.generate/encode`；
- `api_router.py`：FastAPI APIRouter，定义 HTTP 端点并把请求转给 serving 类。

### 2.2 中间件（[build_app L256-311](../../vllm/entrypoints/openai/api_server.py#L256)）

| 中间件 | 作用 |
|--------|------|
| `CORSMiddleware` | 跨域，由 `--allowed-origins` 等配置 |
| `AuthenticationMiddleware` | `--api-key`/`VLLM_API_KEY` 鉴权 |
| `XRequestIdMiddleware` | `--enable-request-id-headers`，注入/回传 X-Request-Id |
| `ScalingMiddleware` | 弹性扩缩容状态检查 |
| `WebSocketMetricsMiddleware` | realtime 端点的 WebSocket 指标 |
| `log_response`（debug） | `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` 打印响应 |
| 自定义 `--middleware` | 用 `"module:object"` 动态加载用户中间件 |

### 2.3 异常处理

把 `HTTPException`、`RequestValidationError`、`EngineGenerateError`、`EngineDeadError`、`GenerationError` 和兜底 `Exception` 分别映射成统一 JSON 错误响应（`http_exception_handler` 等）。

### 2.4 lifespan

FastAPI 的 `lifespan` 上下文在启动时通过 `init_app_state` 初始化所有 serving 组件，关闭时优雅关停 engine。

---

## 3. 应用状态：init_app_state

[init_app_state](../../vllm/entrypoints/openai/api_server.py#L317) 把引擎客户端和各种 serving 服务挂到 `app.state`：

- `state.engine_client`：AsyncLLM 实例；
- `state.vllm_config`、`state.args`；
- `state.openai_serving_models`（`OpenAIServingModels`）：模型列表、LoRA 模块、model registry；
- `state.openai_serving_render`（`OpenAIServingRender`）：chat template 渲染、tool 解析、reasoning parser；
- `state.openai_serving_tokenization`（`OpenAIServingTokenization`）：`/tokenize`、`/detokenize`；
- 再按任务调用各模块的 `init_*_state`，如 `init_generate_state`（创建 chat/completions 的 serving 实例）、`init_generative_scoring_state`、`init_transcription_state`、`init_realtime_state`、`init_pooling_state`。

这一步还会处理：
- `served_model_name`（可为多个别名）；
- chat template 加载（`load_chat_template`，支持文件/内置/请求自带）；
- LoRA 静态模块（`process_lora_modules`，含多模态默认 LoRA）；
- 把引擎侧的 `enable_in_reasoning` 标志传播到 API 进程（因为两端是不同进程，contextvar 不共享）。

---

## 4. AsyncLLM：API 进程里的引擎门面

[AsyncLLM](../../vllm/v1/engine/async_llm.py#L70) 继承 `EngineClient`（[engine/protocol.py:40](../../vllm/engine/protocol.py#L40)），是 API 层与引擎之间的异步接口。

### 4.1 构造时创建的组件

[async_llm.py:73-200](../../vllm/v1/engine/async_llm.py#L73)：

```python
self.renderer = renderer_from_config(vllm_config)      # chat template / tokenizer 渲染
self.input_processor = InputProcessor(vllm_config, renderer)
self.output_processor = OutputProcessor(renderer.tokenizer, stream_interval, ...)
self.engine_core = EngineCoreClient.make_async_mp_client(...)  # ZMQ 连 EngineCore
self.logger_manager = StatLoggerManager(...)          # 指标/日志
self._run_output_handler()                            # 启动输出处理后台任务
```

- **InputProcessor**（`input_processor.py`）：把高层 `Request`/`SamplingParams` 转成 `EngineCoreRequest`，处理 prompt tokenize、多模态占位符、LoRA、prefix cache salt 等。
- **OutputProcessor**（`output_processor.py`）：把 `EngineCoreOutputs` 转成用户可见的 `RequestOutput`，做 detokenize、logprobs、finish reason、累计统计。
- **EngineCoreClient**（`core_client.py`）：`make_async_mp_client` 产出的是多进程客户端，内部有输入/输出 ZMQ 队列；同步方法（`add_request`/`get_output`）和异步版本（`*_async`）成对存在。
- **output_handler**：一个 asyncio Task（`_run_output_handler`，[L637](../../vllm/v1/engine/async_llm.py#L637)）循环 `get_output_async()`，把引擎输出分发给各请求的 `RequestStream`，由 OutputProcessor 处理后投递给 API 端点的生成器。

### 4.2 关键方法

| 方法 | 作用 |
|------|------|
| `generate(request, ...)`（[L524](../../vllm/v1/engine/async_llm.py#L524)） | 提交请求并返回 `RequestStream`，端点 `async for` 它产出 `RequestOutput` |
| `encode(...)`（[L801](../../vllm/v1/engine/async_llm.py#L801)） | 池化模型（embedding/score/classify）入口 |
| `add_request` / `abort` | 增删请求 |
| `start_profile/stop_profile/reset_*_cache` | 透传到 EngineCore |
| `sleep/wake_up/pause_generation/resume_generation` | 引擎生命周期控制 |
| `add_lora/remove_lora/pin_lora/list_loras` | LoRA 管理 |
| `collective_rpc` | 在所有 worker 上调用方法（如重启、调试） |
| `scale_elastic_ep` | 弹性专家并行扩缩容 |
| `init_weight_transfer_engine/start_weight_update/update_weights/finish_weight_update` | 在线权重更新 |
| `check_health/do_log_stats` | 健康检查与指标 |
| `shutdown` | 关闭 EngineCore 与 output_handler |

`from_vllm_config` / `from_engine_args`（[L203/L232](../../vllm/v1/engine/async_llm.py#L203)）是常用工厂方法。

### 4.3 generate 的内部数据流（预览，第 2 篇详述）

```
API 端点 (chat/completions)
   └─ engine_client.generate(EngineInput, sampling_params)
        ├─ InputProcessor 处理 → Request + EngineCoreRequest
        ├─ RequestStream 建立（future/queue）
        ├─ engine_core.add_request_async(core_req)
        └─ yield RequestOutput（由 output_handler 推送）
```

`generate` 支持 streaming/non-streaming、detokenize 控制、自定义 `request_id` 等，端点通常再把 `RequestOutput` 转成 OpenAI 的 `ChatCompletionChunk`/`Completion` 或 SSE 事件。

---

## 5. HTTP 服务：launcher.serve_http

[serve_http](../../vllm/entrypoints/launcher.py#L26) 用 Uvicorn 启动 FastAPI app：

1. 打印所有路由（GET/POST 路由与 WebSocket 端点）；
2. 从 kwargs 弹出并设置 `h11_max_incomplete_event_size`、`h11_max_header_count`（HTTP 头大小/数量限制）；
3. 建 `uvicorn.Config` + `uvicorn.Server`，`app.state.server = server`；
4. 启动两个后台任务：
   - `server.serve(sockets=[sock])`：真正的 HTTP 服务；
   - `watchdog_loop(server, engine_client)`（[L144](../../vllm/entrypoints/launcher.py#L144)）：监控引擎是否死亡，`terminate_if_errored` 在引擎出错时终止服务；
5. 注册 SIGINT/SIGTERM 处理器；`handle_shutdown` 在收到信号时调 `engine_client.shutdown(timeout=shutdown_timeout)` 并停服；
6. 可选 `SSLCertRefresher`（`--ssl-refresh-time`）热更新证书；
7. 端口被占用时，`find_process_using_port` 打印占用进程信息。

`socket` 可由 `create_server_socket`/`create_server_unix_socket` 预建（支持 TCP 与 Unix domain socket）。

---

## 6. 进程模型总结

### 6.1 单节点多卡（默认）

```
API 进程（uvicorn + AsyncLLM）
   │ ZMQ
   ├─ EngineCoreProc（主，rank 0）
   │     └─ Executor (mp/ray) → Worker 0..N（每 GPU 一个）
   └─ （DP 时）DPEngineCoreProc × DP_size
```

- API 进程负责所有 CPU 密集的预处理/后处理；
- EngineCore 进程持有 Scheduler 与 Executor；
- Worker 进程由 Executor 拉起（`MultiprocExecutor`/`RayDistributedExecutor`）。

### 6.2 数据并行（DP）

`DPEngineCoreProc` 在 EngineCore 层复制多份（每个 DP rank 一个 engine），`EngineCoreClient` 按 `client_count/client_index` 连接多个 engine，按请求/波次分发；API 进程通过 `wave_complete/start_wave` 信号协调（见 `vllm/v1/engine/__init__.py`）。MoE + DP 用 `DPMoEEngineCoreActor`/`DPEngineCoreProc`，非 MoE 用普通版本。

### 6.3 Ray 部署

`EngineCoreActor`/`DPMoEEngineCoreActor`（[core.py:2098/2121](../../vllm/v1/engine/core.py#L2098)）把 EngineCore 包成 Ray actor，`EngineCoreActorMixin` 处理可见设备设置与握手；API 进程仍在前端，通过 ZMQ 连到 Ray actor 内的 engine。

### 6.4 远程 engine（API 与引擎分离）

通过 `build_async_engine_client` 的远程分支，API 进程可以不启动本地 EngineCore，而是连接已在别处运行的 engine（配合 PD 分离、多前端副本）。

---

## 7. 与之前篇章的衔接

- **第 04 篇 Worker/ModelRunner**：EngineCore 进程内 `step()` 调的就是它们，产出 `EngineCoreOutputs`；
- **第 06 篇模型实现 / 第 05 篇 layers**：在 Worker 进程被加载执行；
- **第 08 篇分布式**：Executor/Worker 的通信组；
- 本篇的 AsyncLLM 正是把 HTTP 请求翻译成第 04 篇能消费的 `EngineCoreRequest`，并把其 `ModelRunnerOutput` 翻译回 HTTP 响应。

---

## 小结

1. **API Server 位于 engine 之前**，是独立的 FastAPI/Uvicorn 进程，通过 ZMQ 与独立的 EngineCore 进程通信；
2. `run_server → build_async_engine_client → build_app → init_app_state → launcher.serve_http` 是启动主线；
3. `build_app` 按 `supported_tasks` 装配 OpenAI/管理/池化/实时等路由与中间件；
4. `AsyncLLM` 是 API 层的引擎门面，组合 `InputProcessor`、`OutputProcessor`、`EngineCoreClient`、logger 和 output_handler 后台任务；
5. `serve_http` 负责 Uvicorn 生命周期、信号处理、看门狗与 SSL 热更新；
6. 进程模型支持单 engine、DP 多 engine、Ray actor、远程 engine 等多种拓扑。

下一篇 [API Server 之二：请求处理流水线](./api-server-02-请求流水线.md) 将深入 `AsyncLLM.generate`、`InputProcessor`、流式输出、`OutputProcessor` 与 detokenizer，看一个 `/v1/chat/completions` 请求如何在 API 进程里流动。
