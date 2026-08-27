# Tracing 之三：上下文传播与埋点实践

> 相关源码：`vllm/entrypoints/openai/engine/serving.py`、`vllm/v1/engine/{async_llm,core_client,output_processor}.py`、`vllm/tracing/`

前两篇讲了 tracing 的 API 与 OTel 后端。本篇聚焦端到端实践：trace header 如何从 HTTP 一路传到 GPU worker、关键 span 的设计、属性如何填写，以及如何本地部署 collector 查看 trace。

---

## 1. Trace 上下文的完整传播链

vLLM 里有两种传播机制协同工作：
1. **显式传播**：API 层把 `traceparent` 提取出来，放进 `EngineCoreRequest.trace_headers`，跨 ZMQ 传到 EngineCore；
2. **环境变量传播**：`propagate_trace_to_env` 把当前 span 注入 `os.environ`，供 spawn 出的 Worker 子进程继承。

### 1.1 API 层提取 header

[OpenAIServing._get_trace_headers](../../vllm/entrypoints/openai/engine/serving.py#L582)：

```python
async def _get_trace_headers(self, headers):
    is_tracing_enabled = await self.engine_client.is_tracing_enabled()
    if is_tracing_enabled:
        return extract_trace_headers(headers)   # 只取 traceparent/tracestate
    if contains_trace_headers(headers):
        log_tracing_disabled_warning()          # 带了 header 但没启用 tracing，告警一次
    return None
```

各 serving 端点（chat/completions/responses/pooling/scoring）在调 `engine_client.generate` 前取一次 header，作为 `trace_headers` 参数传入。

### 1.2 跨进程传到 EngineCore

`AsyncLLM.generate/add_request` 接收 `trace_headers`（[async_llm.py:291](../../vllm/v1/engine/async_llm.py#L291)），写入 `EngineCoreRequest.trace_headers`（[__init__.py:123](../../vllm/v1/engine/__init__.py#L123)），随 ZMQ 发给 EngineCore 进程。EngineCore 处理时把它放进输出：

- `EngineCoreOutput.trace_headers`（[__init__.py:203](../../vllm/v1/engine/__init__.py#L203)）携带同一批 header 回传。

这样即使 API 进程与 EngineCore 进程不共享内存/环境，trace 上下文也不丢失。

### 1.3 OutputProcessor 还原上下文

请求结束时，[output_processor.do_tracing](../../vllm/v1/engine/output_processor.py#L731)：

```python
trace_context = extract_trace_context(engine_core_output.trace_headers)
...
instrument_manual(
    span_name="llm_request",
    start_time=arrival_time_ns,
    attributes=attributes,
    context=trace_context,       # 用回传的 header 还原父上下文
    kind=SpanKind.SERVER,
)
```

`llm_request` 这个 SERVER span 因此挂在外部调用方的 trace 下，它的属性记录了完整的延迟分解（见第 2 节）。

### 1.4 Worker 子进程的环境继承

EngineCore 内被 `@instrument` 装饰的函数执行时，`propagate_trace_to_env()` 把当前 trace 注入环境变量；MultiprocExecutor spawn worker 时子进程继承环境。Worker 里 `maybe_init_worker_tracer` 建好 provider 后，`@instrument` 的 `_get_smart_context()` 从 `os.environ` 读到 `traceparent`，自动把 worker span 接到父 span。

> 因此一条 HTTP 请求的 trace 能跨越 API → EngineCore → Worker 三个进程连成一棵树。

---

## 2. 关键 Span 与属性设计

### 2.1 请求根 span：`llm_request`

由 `instrument_manual` 在 OutputProcessor 里创建，是每个请求的服务端根 span（`SpanKind.SERVER`），父上下文来自 HTTP header。它携带：

**延迟（秒）**
- `gen_ai.latency.e2e`
- `gen_ai.latency.time_to_first_token`
- `gen_ai.latency.time_in_queue`
- `gen_ai.latency.time_in_model_prefill`
- `gen_ai.latency.time_in_model_decode`
- `gen_ai.latency.time_in_model_inference`

**用量**
- `gen_ai.usage.prompt_tokens`
- `gen_ai.usage.completion_tokens`

**请求参数**
- `gen_ai.request.id`、`gen_ai.request.max_tokens`、`gen_ai.request.temperature`、`gen_ai.request.top_p`、`gen_ai.request.n`

这些来自 `RequestState.stats`（queued_ts/scheduled_ts/first_token_ts/last_token_ts）。

### 2.2 阶段 span（@instrument）

各子系统用默认 span 名（函数 `__qualname__`）和 code 属性自动埋点：

| 阶段 | 代表函数/文件 | 观察内容 |
|------|--------------|---------|
| 引擎调度 | `EngineCore.step`、`_process_input_queue`（core.py） | 每步耗时、输入处理 |
| 跨进程通信 | `EngineCoreClient` 方法（core_client.py） | ZMQ 收发开销 |
| 模型执行 | `GPUModelRunner.execute_model`、CPU model runner | GPU forward 时间 |
| Worker | `GPUWorker.execute_method`、worker_base | 方法分发、collective_rpc |
| Executor | `abstract.py`、`multiproc_executor.py` | 执行器调度 |
| 模型加载 | `base_loader/default_loader/weight_utils` | 权重加载各阶段 |
| 编译 | `VllmBackend.__call__`、Dynamo transform（backends.py） | 编译耗时，含 manual span |
| 预热 | `deep_gemm_warmup.py` | kernel 预热 |
| HTTP 入口 | `api_server.py` | 请求进入 |

code 属性（`code.function`/`code.namespace`/`code.filepath`/`code.lineno`）让你在后端能直接跳转到源码位置。

### 2.3 编译期 manual span

[compilation/backends.py:1156](../../vllm/compilation/backends.py#L1156)：

```python
instrument_manual("Dynamo bytecode transform", start_time, None, attributes)
```

编译发生在请求之外、没有自然的"函数作用域"，用 manual span 记录一段历史耗时。

---

## 3. 自定义埋点

### 3.1 给函数加 span

```python
from vllm.tracing import instrument

@instrument(span_name="my_stage", attributes={"component": "x"})
async def my_async_fn(...):
    ...

@instrument
def my_sync_fn(...):   # span 名默认 my_sync_fn
    ...
```

- 自动支持 sync/async；
- tracing 未启用时原样返回函数，零开销；
- 异常默认记录到 span（`record_exception=False` 可关）。

### 3.2 手动记录一段耗时

```python
from vllm.tracing import instrument_manual, SpanKind
import time

start_ns = time.perf_counter_ns()
# ... 做一段事 ...
instrument_manual(
    span_name="my_phase",
    start_time=start_ns,
    attributes={"gen_ai.request.id": rid},
    kind=SpanKind.INTERNAL,
)
```

适合无法用 `with` 包裹的场景，或起止时间跨越异步边界。

### 3.3 保护昂贵逻辑

```python
from vllm.tracing import is_tracing_available, extract_trace_headers

if is_tracing_available():
    headers = extract_trace_headers(raw_request.headers)
```

### 3.4 自定义属性

用 `SpanAttributes` 常量命名，避免与 OTel 约定冲突；自定义属性建议加前缀（如 `vllm.*`、`gen_ai.*`）。

---

## 4. 启用方式

### 4.1 启动参数

```bash
pip install opentelemetry-api opentelemetry-sdk \
            opentelemetry-exporter-otlp-proto-grpc

vllm serve meta-llama/Llama-3-8B-Instruct \
  --otlp-traces-endpoint http://localhost:4317/v1/traces
```

- gRPC 默认（端口 4317）；
- HTTP/protobuf：设 `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`，endpoint 通常是 `http://localhost:4318/v1/traces`。

### 4.2 环境变量

vLLM 用标准 OTel 环境变量：
- `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`（vLLM 启动时自动写入，供子进程读）；
- `OTEL_EXPORTER_OTLP_PROTOCOL`；
- 也可设 `OTEL_SERVICE_NAME`、`OTEL_RESOURCE_ATTRIBUTES` 等。

### 4.3 客户端发起带 trace 的请求

用任意 W3C Trace Context 兼容的客户端（OpenTelemetry instrumentation、curl 手动加头）：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01" \
  -H "tracestate: congo=t61rcWkgMzE" \
  -d '{"model":"...","messages":[{"role":"user","content":"hi"}]}'
```

这样 vLLM 产生的 span 会挂在这个外部 trace 下。

---

## 5. 本地验证（Jaeger）

最小化 docker-compose：

```yaml
# docker-compose.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:1
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "4317:4317"     # OTLP gRPC
      - "16686:16686"   # Web UI
```

```bash
docker compose up -d
vllm serve <model> --otlp-traces-endpoint http://localhost:4317/v1/traces
# 发几个请求后打开 http://localhost:16686
```

在 Jaeger 里选择服务（`vllm.llm_engine` / `vllm.engine_core`），可以看到：
- 一条 trace 含 `llm_request` 根 span；
- 下面是 `EngineCore.step`、`GPUModelRunner.execute_model` 等子 span，跨进程连成树；
- 每个 span 的属性含 token 数、延迟分解、code 位置。

生产中通常部署一个 OTel Collector，再转发到 Jaeger/Tempo/云厂商追踪服务。

---

## 6. 注意事项与排错

1. **没装 OTel 包**：`is_tracing_available()` 为 False，所有 API 变 no-op；`otel_import_error_traceback` 记录了原始 ImportError。
2. **子进程没有 span**：确认 endpoint 通过环境变量传播（`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`），且 worker 是在某个被装饰函数上下文中 spawn 的。
3. **trace 断链**：检查 `traceparent` 头是否被代理剥离；用 `contains_trace_headers`/`extract_trace_headers` 确认。
4. **收到 header 但未启用**：日志会有一次性 "Received a request with trace context but tracing is disabled" 警告。
5. **span 未导出**：进程退出时 `atexit` 会 flush BatchSpanProcessor；异常 kill 可能丢最后一批，可配置更小的批处理间隔。
6. **性能**：tracing 有少量开销，`is_tracing_available` 守卫可避免无谓计算；BatchSpanProcessor 是异步的，正常负载影响很小。
7. **多线程**：OTel context 是 contextvar 本地的；线程池里需要显式 `attach` context 或用 `trace.get_current_span`。

---

## 7. 扩展新后端

`_REGISTERED_TRACING_BACKENDS` 预留了扩展点（第 1 篇）。新增后端只需提供 5 个函数：

```python
def is_mybackend_available() -> bool: ...
def init_mybackend_tracer(name, endpoint, extra): ...
def init_mybackend_worker_tracer(name, kind, pname): ...
def instrument_mybackend(func, span_name, attrs, record_exc): ...
def manual_instrument_mybackend(name, start, end, attrs, ctx, kind): ...

_REGISTERED_TRACING_BACKENDS["mybackend"] = (
    is_mybackend_available, init_mybackend_tracer, init_mybackend_worker_tracer,
    instrument_mybackend, manual_instrument_mybackend,
)
```

门面函数目前硬编码取 `"otel"`，若要支持多后端可改为按配置选择。

---

## 小结

1. **上下文双轨传播**：HTTP header → `trace_headers` 字段跨 ZMQ 到 EngineCore（显式），当前 span → 环境变量供 spawn 的 worker 继承（隐式）；
2. `llm_request` 手动 span 是请求根，记录完整延迟分解与用量；`@instrument` 自动覆盖调度/执行/加载/编译等阶段；
3. 自定义埋点用 `@instrument`/`instrument_manual`，属性用 `SpanAttributes` 常量；
4. 启用只需 `--otlp-traces-endpoint` + 装 OTel 包；客户端传 `traceparent` 即可把 vLLM 纳入分布式 trace；
5. 本地用 Jaeger all-in-one 即可验证，生产经 OTel Collector 转发；
6. tracing 未启用时全部降级为 no-op，开销可控。

至此 Tracing 系列三篇完成，覆盖了从公共 API、OTel 后端实现到端到端上下文传播与部署实践的全部主线。
