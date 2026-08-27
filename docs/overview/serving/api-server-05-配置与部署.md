# API Server 之五：配置与部署

> 相关源码：`vllm/entrypoints/openai/cli_args.py`、`vllm/entrypoints/openai/api_server.py`、`vllm/entrypoints/launcher.py`、`vllm/entrypoints/ssl.py`、`vllm/entrypoints/logger.py`、`vllm/entrypoints/openai/server_utils.py`

本篇汇总 API Server 的配置项分类、中间件、SSL/鉴权、日志指标、Uvicorn 部署选项，以及生产部署拓扑。

---

## 1. CLI 参数体系

`vllm serve` 的参数由若干 dataclass 声明式定义，`make_arg_parser`（[cli_args.py:329](../../vllm/entrypoints/openai/cli_args.py#L329)）把它们注册到 argparse：

- `FrontendArgs`（[L225](../../vllm/entrypoints/openai/cli_args.py#L225)）：API Server 自身的参数（本篇重点）；
- `AsyncEngineArgs`：引擎参数（模型路径、TP/PP、KV cache、量化、调度等，见引擎相关篇章）；
- `*Args`：LoRA、多模态、观测、speculative、池化等子配置。

`@config` 装饰器把 dataclass 字段转成 `--kebab-case` CLI 参数。

### 1.1 网络与监听（FrontendArgs）

| 参数 | 默认 | 作用 |
|------|------|------|
| `--host` | None | 监听地址（None 时用配置/默认） |
| `--port` | 8000 | HTTP 端口 |
| `--uds` | None | Unix domain socket 路径（设置后忽略 host/port） |
| `--root-path` | None | 反向代理路径前缀（FastAPI root_path） |

### 1.2 CORS

- `--allowed-origins`（默认 `["*"]`）、`--allowed-methods`、`--allowed-headers`：配置 `CORSMiddleware`，用 JSON 解析；
- `--allow-credentials`：是否允许凭证。

### 1.3 鉴权

- `--api-key`：可重复指定多个；也可用环境变量 `VLLM_API_KEY`。CLI 值优先。设置后 `AuthenticationMiddleware` 校验 `Authorization: Bearer <key>` 或 `api-key` 头。

### 1.4 SSL/TLS

| 参数 | 作用 |
|------|------|
| `--ssl-keyfile` | 私钥文件 |
| `--ssl-certfile` | 证书文件 |
| `--ssl-ca-certs` | CA 证书（校验客户端） |
| `--ssl-cert-reqs` | 是否要求客户端证书（`ssl.CERT_NONE/REQUIRED`） |
| `--ssl-ciphers` | TLS1.2 及以下密码套件 |
| `--enable-ssl-refresh` | 证书文件变化时热刷新 SSL Context（`SSLCertRefresher`） |

### 1.5 文档与请求头

- `--disable-fastapi-docs`：关闭 `/openapi.json`、`/docs`、`/redoc`；
- `--enable-offline-docs`：用内置静态资源支持离线文档；
- `--enable-request-id-headers`：注入/回传 `X-Request-Id`；
- `--middleware`：可重复，`module:object` 形式加载自定义 ASGI 中间件（类用 `add_middleware`，函数用 `@app.middleware("http")`）。

### 1.6 HTTP 解析限制

- `--h11-max-incomplete-event-size`（默认 4MB）：防大包滥用；
- `--h11-max-header-count`（默认 256）：防头数量滥用。

### 1.7 日志

- `--uvicorn-log-level`（info）、`--disable-uvicorn-access-log`；
- `--disable-access-log-for-endpoints`：逗号分隔，屏蔽高频端点（如 `/health`、`/metrics`）的访问日志；
- `--log-config`：自定义 uvicorn 日志配置（`server_utils.load_log_config`）。

### 1.8 其他

- `--served-model-name`：模型别名（可多个）；
- `--chat-template`、`--chat-template-content-format`、`--trust-request-chat-template`、`--default-chat-template-kwargs`：模板控制；
- `--tool-call-parser`、`--enable-auto-tool-choice`、`--structured-outputs` 等：工具/结构化输出；
- `--lora-modules`：启动时加载静态 LoRA；
- `--max-log-len`、`--enable-log-requests`：请求日志（`RequestLogger`）；
- `--enable-server-load-tracking`：暴露负载指标；
- `--enable-flash-late-interaction`：在 API 进程 GPU 上跑 late-interaction 打分。

引擎参数（TP/PP/量化/`--gpu-memory-utilization`/`--max-model-len`/`--kv-cache-dtype` 等）属于 `AsyncEngineArgs`，透传给 AsyncLLM/EngineCore。

---

## 2. 中间件与异常处理

[build_app](../../vllm/entrypoints/openai/api_server.py#L157) 装配顺序（外→内）：

1. `CORSMiddleware`；
2. 异常处理器（`HTTPException`/`RequestValidationError`/`EngineGenerateError`/`EngineDeadError`/`GenerationError`/兜底 `Exception`）；
3. `AuthenticationMiddleware`（若配置 api key）；
4. `XRequestIdMiddleware`（若启用）；
5. `ScalingMiddleware`（弹性扩缩容状态检查）；
6. `WebSocketMetricsMiddleware`（realtime）；
7. 可选 `log_response`（`VLLM_DEBUG_LOG_API_SERVER_RESPONSE`，含敏感信息，生产勿用）；
8. 用户自定义 `--middleware`。

### AuthenticationMiddleware（[server_utils.py:38](../../vllm/entrypoints/openai/server_utils.py#L38)）

从 `Authorization` 头（`Bearer ` 前缀）或 `x-api-key`/`api-key` 取 token，与配置的任一 key 比对；失败返回 401。

### XRequestIdMiddleware（[L89](../../vllm/entrypoints/openai/server_utils.py#L89)）

若请求带 `X-Request-Id` 则透传，否则生成一个，写入 `request.state.request_id` 并回传响应头。

### 异常响应

所有错误统一成 OpenAI 风格 `{"error": {"message", "type", "code", "param"}}`，带正确 HTTP 状态码（400/404/422/500 等）。

---

## 3. SSL 证书热刷新

[ssl.py:15](../../vllm/entrypoints/ssl.py#L15) 的 `SSLCertRefresher` 在 `enable_ssl_refresh` 时后台轮询证书文件的 mtime，变化时重新加载 SSL context 并替换到运行中的 uvicorn server，无需重启即可续期证书。

---

## 4. 日志与请求追踪

- **Uvicorn 访问日志**：`get_uvicorn_log_config`（[server_utils.py:133](../../vllm/entrypoints/openai/server_utils.py#L133)）配置格式，可按端点屏蔽；
- **RequestLogger**（[logger.py:17](../../vllm/entrypoints/logger.py#L17)）：`--enable-log-requests` 时记录请求/响应，`max_log_len` 截断；
- **流式响应日志**：`_log_streaming_response`/`_log_non_streaming_response`、`SSEDecoder` 用于在日志里解码 SSE 内容；
- **Tracing**：`--otlp-traces-endpoint` 等把请求与引擎 span 关联（`_get_trace_headers`），`RequestResponseMetadata` 挂在 `raw_request.state` 上；
- **Prometheus 指标**：`orca_metrics.py` 与 `instrumentator` 暴露 `/metrics`（含每请求、每引擎统计）。

---

## 5. Uvicorn 运行与生命周期

[launcher.serve_http](../../vllm/entrypoints/launcher.py#L26)：

- 启动时打印所有路由表；
- 创建 `uvicorn.Config`/`Server`，设置 h11 限制；
- `app.state.server = server`；
- 并发跑 `server.serve(sockets=[sock])` 与 `watchdog_loop`；
- 注册 SIGINT/SIGTERM，`handle_shutdown` 调 `engine_client.shutdown(timeout=shutdown_timeout)` 后停服；
- 可选 SSL refresher；
- 端口占用时 `find_process_using_port` 打印占用进程命令行。

socket 由 `create_server_socket`（TCP）或 `create_server_unix_socket`（UDS）预建，支持 SO_REUSEADDR 与权限设置。`run_server_worker` 负责把这些串起来。

### lifespan

FastAPI 的 `lifespan`（[server_utils.py:447](../../vllm/entrypoints/openai/server_utils.py#L447)）在启动时 `init_app_state`（第 1 篇），关闭时取消后台任务、关闭 engine。

---

## 6. 部署拓扑

### 6.1 单机单引擎（默认）

```
客户端 → API Server(API 进程) → ZMQ → EngineCore 进程 → Workers(GPU)
```

`--tensor-parallel-size N` 让 Executor 拉起 N 个 worker；`--pipeline-parallel-size` 切流水。

### 6.2 数据并行（多引擎副本）

- `--data-parallel-size` / `--data-parallel-size-local` 启动多个 EngineCore；
- API 进程通过 `EngineCoreClient` 连接多个 engine，按请求/波次分发；
- `--enable-expert-parallel` 等影响 MoE 布局。

### 6.3 Ray 集群

`vllm serve ... --distributed-executor-backend ray`，EngineCore/Actor 作为 Ray actor 跨节点调度，API 进程可在前端节点。

### 6.4 多前端 + 远程 Engine

API 进程可不内置引擎，连接已运行的 engine（`build_async_engine_client` 的远程分支），从而水平扩展无状态的 API 副本。

### 6.5 Prefill-Decode 分离

- KV connector 配置（第 08 篇）做 KV 传输；
- `serve/disagg` 端点或外部路由器协调 P/D 实例；
- 可配合 `--kv-transfer-config`。

### 6.6 反向代理

API Server 常位于 Nginx/Envoy 之后：设置 `--root-path`、信任的 `X-Request-Id`、CORS；用 UDS 或内网端口通信；用 `--api-key` 做边界鉴权。

### 6.7 容器/K8s

- 用 `--host 0.0.0.0`；
- readiness/liveness 探针打 `/health`（模型加载完成才就绪）；
- `--port`/`--uds` 与 service 对齐；
- 弹性扩缩容可调用 elastic_ep/RLHF/sleep 等管理 API（第 4 篇）。

---

## 7. 性能与运维建议

- 生产环境开启 `--api-key`、关闭 docs（`--disable-fastapi-docs`）、设置 CORS 白名单；
- 高频健康检查端点用 `--disable-access-log-for-endpoints` 降噪；
- 用 `--uds` 替代 TCP 做同机反代可降低开销；
- `--h11-max-header-count`/`max-incomplete-event-size` 按客户端行为收紧；
- 指标与 tracing 按需开启，避免 `/metrics` 本身成为热点；
- 多模型分时复用 GPU 用 sleep/wake API；
- 证书续期用 `--enable-ssl-refresh`；
- 自定义中间件用 `--middleware module:object` 注入（注意它在 vLLM 中间件外层）。

---

## 小结

1. **参数由 `FrontendArgs`/`AsyncEngineArgs` 等 dataclass 声明式定义**，覆盖网络、CORS、鉴权、SSL、日志、文档、中间件等；
2. **中间件链顺序固定**：CORS→异常→鉴权→请求 ID→扩缩容→WS 指标→自定义；
3. **SSL 支持热刷新证书**，Uvicorn 由 `launcher.serve_http` 管理生命周期与信号；
4. **日志/追踪/指标**贯穿请求，支持按端点屏蔽与 OpenTelemetry；
5. **部署形态**从单机到 DP/Ray/多前端/PD 分离均可通过参数组合实现，API 进程与 EngineCore 进程始终解耦。

至此 API Server 系列五篇完成，从启动架构、请求流水线、OpenAI 接口、批处理/扩展到配置部署，覆盖了 `vllm/entrypoints/` 与 `vllm/v1/engine/` 中 API 相关的全部主线。
