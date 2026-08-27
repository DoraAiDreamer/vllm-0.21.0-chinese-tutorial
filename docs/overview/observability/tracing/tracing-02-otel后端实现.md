# Tracing 之二：OpenTelemetry 后端实现

> 源码：`vllm/tracing/otel.py`

本篇逐段解析 `otel.py`——vLLM 目前唯一的 tracing 后端。它负责初始化 OTel TracerProvider、创建/导出 span、包装同步/异步函数，以及跨进程上下文传播。

---

## 1. 依赖探测（软依赖）

[otel.py:18-53](../../vllm/tracing/otel.py#L18) 用一个大 `try/except ImportError` 导入 OpenTelemetry 各组件：

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpExporter
from opentelemetry.propagate import inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
```

导入成功时 `_IS_OTEL_AVAILABLE = True`；失败则把 `trace/Tracer/...` 置为 `Any`/`None`，并记录 `otel_import_error_traceback`，让没有安装 OTel 的环境仍能 import vLLM。`is_otel_available()` 对外暴露这个开关。

> 这是整个 tracing 子系统"零开销降级"的基础——所有门面函数在调用前检查 `_IS_OTEL_AVAILABLE`。

---

## 2. 初始化 TracerProvider

### 2.1 init_otel_tracer

[init_otel_tracer](../../vllm/tracing/otel.py#L60) 在主进程建立 OTel SDK：

```python
def init_otel_tracer(instrumenting_module_name, otlp_traces_endpoint, extra_attributes=None):
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = otlp_traces_endpoint  # 让子进程继承
    resource_attrs = {
        "vllm.instrumenting_module_name": instrumenting_module_name,
        "vllm.process_id": str(os.getpid()),
    }
    if extra_attributes:
        resource_attrs.update(extra_attributes)
    resource = Resource.create(resource_attrs)

    trace_provider = TracerProvider(resource=resource)
    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    atexit.register(trace_provider.shutdown)
    return trace_provider.get_tracer(instrumenting_module_name)
```

关键点：
- **Resource**：标识这一路 trace 来自哪个服务/进程（`service.name` 默认是 `instrumenting_module_name`，外加 pid 与自定义属性）；
- **OTLP exporter**：通过 gRPC 或 HTTP 把 span 发到 collector；
- **BatchSpanProcessor**：批量异步导出，避免阻塞推理；
- **`set_tracer_provider`**：设为全局默认 provider，之后 `trace.get_tracer(__name__)` 任意处可取；
- **endpoint 写入环境变量**：子进程（EngineCore/Worker）通过它继承配置；
- **`atexit.register(shutdown)`**：进程退出时 flush 剩余 span。

### 2.2 协议选择：get_span_exporter

[get_span_exporter](../../vllm/tracing/otel.py#L94)：

```python
protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
if protocol == "grpc":
    return OTLPGrpcExporter(endpoint=endpoint, insecure=True)
elif protocol == "http/protobuf":
    return OTLPHttpExporter(endpoint=endpoint)
```

通过标准 OTel 环境变量 `OTEL_EXPORTER_OTLP_PROTOCOL` 选择，默认 gRPC（`insecure=True` 适配本地/内网 collector）。

### 2.3 Worker 子进程初始化

[init_otel_worker_tracer](../../vllm/tracing/otel.py#L105)：

```python
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
if not otlp_endpoint:
    return None
extra_attrs = {
    "vllm.process_kind": process_kind,   # "engine_core" / "worker"
    "vllm.process_name": process_name,
}
return init_otel_tracer(instrumenting_module_name, otlp_endpoint, extra_attrs)
```

子进程不接收 endpoint 参数，而是从环境变量读主进程写入的值；带上 `process_kind/process_name` 属性，后端可据此筛选。若环境里没有 endpoint，直接返回 None（不追踪）。

---

## 3. Span 的包装：instrument_otel

[instrument_otel](../../vllm/tracing/otel.py#L134) 是 `@instrument` 的实际实现。

### 3.1 预计算静态 code 属性

```python
code_attrs = {
    LoadingSpanAttributes.CODE_FUNCTION: func.__qualname__,
    LoadingSpanAttributes.CODE_NAMESPACE: func.__module__,
    LoadingSpanAttributes.CODE_FILEPATH: func.__code__.co_filename,
    LoadingSpanAttributes.CODE_LINENO: str(func.__code__.co_firstlineno),
}
```

这些在装饰时算一次（每次调用不必再算），与用户传入的 `attributes` 合并。span 名默认用 `func.__qualname__`。

### 3.2 同步/异步两个 wrapper

```python
@functools.wraps(func)
async def async_wrapper(*args, **kwargs):
    tracer = trace.get_tracer(module_name)
    ctx = _get_smart_context()
    with (tracer.start_as_current_span(final_span_name, context=ctx,
                                      attributes=code_attrs,
                                      record_exception=record_exception),
          propagate_trace_to_env()):
        return await func(*args, **kwargs)

@functools.wraps(func)
def sync_wrapper(*args, **kwargs):
    ...  # 同样的 with 块，return func(...)

return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
```

要点：
- 用 `inspect.iscoroutinefunction` 自动选择异步/同步包装，所以 `@instrument` 对两者都正确；
- `start_as_current_span` 把 span 设为当前 span，函数内部的子 span 自动挂到它下面；
- `record_exception=False` 时异常不会记录到 span（默认 True）；
- **`propagate_trace_to_env()`** 是跨进程传播的关键（第 5 节）。

---

## 4. 手动 Span：manual_instrument_otel

[manual_instrument_otel](../../vllm/tracing/otel.py#L183)：

```python
tracer = trace.get_tracer(__name__)
ctx = context if context is not None else _get_smart_context()
span_kwargs = {"name": span_name, "context": ctx, "start_time": start_time}
if kind is not None:
    span_kwargs["kind"] = kind
span = tracer.start_span(**span_kwargs)
if attributes:
    span.set_attributes(attributes)
span.end(end_time=end_time)  # None 时立即结束
```

与装饰器不同：
- **不包裹函数**，直接创建并结束 span；
- 接收 `start_time`/`end_time`（纳秒），用于事后记录一段历史耗时（如请求从到达到完成）；
- 支持 `SpanKind`（SERVER/CLIENT/PRODUCER/CONSUMER/INTERNAL）；
- `llm_request` span 用 `SpanKind.SERVER` 表示它是整个请求的根服务端 span。

---

## 5. 上下文传播（核心机制）

分布式追踪最难的部分是让 span 在多进程/多服务间连起来。vLLM 用 W3C Trace Context（`traceparent`/`tracestate`）。

### 5.1 从 HTTP header 提取

[extract_trace_context](../../vllm/tracing/otel.py#L127)：

```python
if _IS_OTEL_AVAILABLE and headers:
    return TraceContextTextMapPropagator().extract(headers)
return None
```

API Server 收到带 `traceparent` 的请求时，把它解析成 `Context`，后续 `llm_request` 手动 span 用它作为父上下文（见第 3 篇）。

### 5.2 智能父上下文：_get_smart_context

[_get_smart_context](../../vllm/tracing/otel.py#L216)：

```python
current_span = trace.get_current_span()
if current_span.get_span_context().is_valid:
    return None           # 本进程已有活跃 span，用它作父（返回 None 即默认）

carrier = {}
if tp := os.environ.get("traceparent", os.environ.get("TRACEPARENT")):
    carrier["traceparent"] = tp
if ts := os.environ.get("tracestate", os.environ.get("TRACESTATE")):
    carrier["tracestate"] = ts
if not carrier:
    carrier = dict(os.environ)
return TraceContextTextMapPropagator().extract(carrier)
```

逻辑：
1. 如果当前进程/线程已经有一个活跃 span（`@instrument` 嵌套调用），直接挂到它下面（返回 `None`）；
2. 否则看环境变量里的 `traceparent/tracestate`——这是父进程通过 `propagate_trace_to_env` 注入的；
3. 都没有则把整个 `os.environ` 当 carrier（容错，大小写或其他传播字段）。

这让 EngineCore/Worker 在**没有显式传 context 参数**的情况下，自动把新 span 接到主进程的 trace 树上。

### 5.3 注入到环境变量：propagate_trace_to_env

[propagate_trace_to_env](../../vllm/tracing/otel.py#L240) 是一个 contextmanager：

```python
original_state = {k: os.environ.get(k) for k in TRACE_HEADERS}
try:
    inject(os.environ)     # OTel 的 inject() 把 traceparent/tracestate 写进 carrier(即 os.environ)
    yield
finally:
    # 恢复原值/删除
```

`instrument_otel` 在每个被装饰函数执行期间 `with propagate_trace_to_env()`：当前 span 的上下文被写进 `os.environ`。这对以下场景至关重要：

- **多进程 Executor**：Worker 进程由 spawn/fork 创建，会继承父进程环境变量；如果在产生子进程时处于某个 span 内，子进程就能从环境变量读到 `traceparent`，把自己的 span 接上去；
- **子任务/线程**：任何从该函数派生的逻辑，读环境变量即可获得上下文。

退出 `with` 块时恢复环境，避免污染。

> 注意：`os.environ` 在多线程下共享，因此这一机制主要服务于**进程派生**时刻；线程内的嵌套调用靠 `get_current_span`（路径 1）而非环境变量。

---

## 6. 完整数据流

一个带 `traceparent` 的 HTTP 请求，跨进程的 span 树如何形成：

```
[API 进程]
  HTTP 进来，header: traceparent=00-<trace_id>-...
  │
  ├─ extract_trace_context(headers) → ctx
  ├─ @instrument 装饰的端点函数
  │     start_as_current_span(context=ctx)   # span 挂到外部 trace
  │     └─ AsyncLLM.generate
  │           └─ 内部各 @instrument span 自动挂当前 span
  │
  │  (产生 EngineCore 子进程时，环境里有 propagate_trace_to_env 注入的 traceparent)
  ▼
[EngineCore 进程]
  maybe_init_worker_tracer() 读 env 的 endpoint，建 provider
  │
  ├─ core.py step() 等被 @instrument 装饰
  │     调用时 _get_smart_context():
  │       本进程无活跃 span → 读 os.environ 的 traceparent → 提取为 ctx
  │       start_as_current_span(context=ctx)  # 接到 API 进程的 span
  │     └─ executor.collective_rpc → [Worker 进程]
  │
[Worker 进程]
  maybe_init_worker_tracer()
  └─ @instrument 的 execute_model/get_model 等
        _get_smart_context 读环境变量 traceparent → 接到同一棵树
```

最终在 Jaeger/Tempo 里能看到一条完整的 trace：HTTP handler → engine step → worker forward → kernel/编译/加载等子 span，跨进程边界由 `traceparent` 串接。

---

## 7. 导出与部署

- **OTLP endpoint**：`--otlp-traces-endpoint http://collector:4317`（gRPC）或 `/v1/traces`（HTTP）；
- **Protocol**：设 `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`（默认）或 `http/protobuf`；
- **Resource 属性**：可通过 `extra_attributes` 加 `service.name`、环境、版本等；
- **Collector**：通常部署一个 OpenTelemetry Collector 接收，再转 Jaeger/Tempo/Zipkin/云厂商；
- **批处理**：`BatchSpanProcessor` 自动批量发送，进程退出时 `atexit` flush。

---

## 小结

1. `otel.py` 用软依赖导入 OTel，不可用时整体降级为 no-op；
2. `init_otel_tracer` 建 TracerProvider + BatchSpanProcessor + OTLP exporter，并把 endpoint 写入环境变量供子进程继承；
3. `init_otel_worker_tracer` 让 EngineCore/Worker 从环境变量读 endpoint 并打上 `process_kind/process_name`；
4. `instrument_otel` 预计算 code 属性，按函数是否协程选择异步/同步 wrapper，用 `start_as_current_span` 自动建父子关系；
5. `manual_instrument_otel` 用显式时间戳事后记录 span（请求排队/E2E 耗时）；
6. **上下文传播靠 W3C `traceparent`/`tracestate`**：HTTP header 提取 → `_get_smart_context` 智能选父 → `propagate_trace_to_env` 注入环境变量供子进程继承。

下一篇 [Tracing 之三：上下文传播与埋点实践](./tracing-03-上下文传播与埋点实践.md) 将看 API 层如何提取/回传 trace header、各关键路径的 span 设计、属性如何填写，以及如何本地搭建 collector 验证。
