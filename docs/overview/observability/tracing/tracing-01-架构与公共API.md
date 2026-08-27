# Tracing 之一：架构与公共 API

> 源码：`vllm/tracing/__init__.py`、`vllm/tracing/utils.py`

`vllm/tracing/` 是 vLLM 的分布式追踪（tracing）抽象层。它把 OpenTelemetry（OTel）封装成极简的几个公共 API，让引擎、Worker、模型加载器、编译器等各处可以统一打 span（跨度），并在 API Server→EngineCore→Worker 的多进程边界传播 trace context。

本篇讲整体架构和公共 API；第 2 篇讲 OTel 后端实现；第 3 篇讲跨进程上下文传播与实际埋点。

---

## 1. 目录与职责

```
vllm/tracing/
├── __init__.py   # 公共门面(facade)：后端注册表 + instrument/init_tracer 等 API
├── otel.py       # OpenTelemetry 后端实现（TracerProvider、span、上下文注入）
└── utils.py      # SpanAttributes/LoadingSpanAttributes 常量、trace header 工具
```

设计目标：
- **对调用方零成本**：没装/没配 OTel 时，`@instrument` 原样返回函数，不产生任何开销；
- **后端可插拔**：用 `_REGISTERED_TRACING_BACKENDS` 字典注册后端，目前只有 `"otel"`，未来可加其他；
- **跨进程透明传播**：主进程的 trace 上下文能自动传到 EngineCore/Worker 子进程；
- **统一属性命名**：用 `SpanAttributes` 常量对齐 OpenTelemetry GenAI 语义约定。

---

## 2. 后端注册表

[__init__.py:46-63](../../vllm/tracing/__init__.py#L46) 定义了一个内部注册表：

```python
_REGISTERED_TRACING_BACKENDS: dict[str, tuple[
    BackendAvailableFunc,      # () -> bool，后端是否可用（依赖是否安装）
    InitTracerFunc,           # 初始化主进程 tracer
    InitWorkerTracerFunc,     # 初始化 worker 子进程 tracer
    InstrumentFunc,           # 装饰函数
    InstrumentManualFunc,     # 手动创建 span
]] = {
    "otel": (is_otel_available, init_otel_tracer, init_otel_worker_tracer,
             instrument_otel, manual_instrument_otel),
}
```

每个后端提供 5 个能力。门面函数只从注册表里取实现并在 `is_available()` 为真时调用——这是典型的"门面 + 策略"模式，调用方完全不需要 `import opentelemetry`。

类型别名（[L41-45](../../vllm/tracing/__init__.py#L41)）：

```python
BackendAvailableFunc = Callable[[], bool]
InstrumentFunc        = Callable[..., Any]
InstrumentManualFunc  = Callable[..., Any]
InitTracerFunc        = Callable[..., Any]
InitWorkerTracerFunc  = Callable[..., Any]
```

---

## 3. 公共 API 总览

从 `vllm.tracing` 导出的全部公共符号（[__init__.py:26-39](../../vllm/tracing/__init__.py#L26)）：

| API | 作用 |
|-----|------|
| `init_tracer(module_name, endpoint, extra_attributes=None)` | 在主进程（API/LLM 引擎）初始化 OTel TracerProvider |
| `maybe_init_worker_tracer(module_name, process_kind, process_name)` | 在 EngineCore/Worker 子进程按需初始化 tracer |
| `@instrument(...)` | 装饰器：自动给函数包一个 span（同步/异步都支持） |
| `instrument_manual(span_name, start_time, ...)` | 用显式时间戳手动创建一个 span（用于排队等耗时统计） |
| `is_tracing_available()` | 是否有任一后端可用，用来包裹昂贵的 tracing 逻辑 |
| `SpanAttributes` / `SpanKind` | span 属性名常量 / OTel span 类型 |
| `extract_trace_context(headers)` | 从 HTTP headers 提取 OTel 上下文 |
| `extract_trace_headers(headers)` | 只提取 trace 相关 header（`traceparent`/`tracestate`） |
| `contains_trace_headers(headers)` | 是否含 trace 上下文 |
| `log_tracing_disabled_warning()` | 收到 trace header 但 tracing 未启用时告警（只打一次） |
| `otel_import_error_traceback` | OTel 导入失败时的错误堆栈（调试用） |

---

## 4. 初始化 API

### 4.1 init_tracer

[__init__.py:66-75](../../vllm/tracing/__init__.py#L66)：

```python
def init_tracer(instrumenting_module_name, otlp_traces_endpoint, extra_attributes=None):
    is_available, init_tracer_fn, *_ = _REGISTERED_TRACING_BACKENDS["otel"]
    if is_available():
        return init_tracer_fn(...)
```

调用点：
- [async_llm.py:116](../../vllm/v1/engine/async_llm.py#L116)：API 进程在创建 `AsyncLLM` 时，若配置了 `--otlp-traces-endpoint` 就 `init_tracer("vllm.llm_engine", endpoint)`；
- [llm_engine.py:67](../../vllm/v1/engine/llm_engine.py#L67)：离线 `LLM` 类同理。

未配置 endpoint 或未装 OTel 时是 no-op。

### 4.2 maybe_init_worker_tracer

[L78-87](../../vllm/tracing/__init__.py#L78)：在 EngineCore/Worker **子进程**里初始化 tracer。它不接收 endpoint，而是读主进程通过环境变量 `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` 传下来的地址（见第 3 篇）。

调用点：
- [core.py:1095](../../vllm/v1/engine/core.py#L1095)：EngineCoreProc 启动时 `maybe_init_worker_tracer("vllm.engine_core", "engine_core", title)`；
- [core.py:2013](../../vllm/v1/engine/core.py#L2013)：Ray actor 版本；
- [multiproc_executor.py:831](../../vllm/v1/executor/multiproc_executor.py#L831)：worker 初始化时。

子进程的 span 会带上 `vllm.process_kind` 和 `vllm.process_name` 属性，便于在后端区分 API/engine_core/worker。

---

## 5. 埋点 API

### 5.1 @instrument 装饰器

[L90-118](../../vllm/tracing/__init__.py#L90)：

```python
def instrument(obj=None, *, span_name="", attributes=None, record_exception=True):
    if obj is None:
        return functools.partial(instrument, span_name=..., attributes=..., ...)
    if is_available():
        return otel_instrument(func=obj, span_name=..., attributes=..., record_exception=...)
    return obj
```

三种用法：

```python
# 1) 直接装饰（span 名用函数限定名）
@instrument
def load_weights(self, weights): ...

# 2) 带参数
@instrument(span_name="model.forward", attributes={"model": "llama"})
async def forward(self, ...): ...

# 3) 注意：装饰器在 import 时求值，若当时 tracing 不可用就原样返回
```

它自动：
- 区分同步/异步函数（OTel 后端返回 `async_wrapper` 或 `sync_wrapper`）；
- 把函数的 `__qualname__`、`__module__`、文件路径、行号作为 code 属性；
- 选择"智能"父上下文（见第 3 篇 `_get_smart_context`）；
- `record_exception=True` 时自动记录异常到 span。

**可用时返回包装函数，不可用时返回原函数**——所以在没装 OTel 的环境里零开销。

### 5.2 instrument_manual

[L121-145](../../vllm/tracing/__init__.py#L121)：

```python
def instrument_manual(span_name, start_time, end_time=None,
                      attributes=None, context=None, kind=None):
```

用于无法用 `with`/装饰器包裹的场景，尤其是**用显式纳秒时间戳记录一段耗时**。典型用法在 [output_processor.py:785](../../vllm/v1/engine/output_processor.py#L785)：

```python
instrument_manual(
    span_name="llm_request",
    start_time=arrival_time_ns,   # 请求到达时刻
    attributes=attributes,        # gen_ai.usage.*, gen_ai.request.* 等
    context=trace_context,        # 从 HTTP header 传播来的根上下文
    kind=SpanKind.SERVER,
)
```

这里 span 在请求结束时一次性创建，start 是到达时刻，从而准确表达"请求在队列+调度+生成中的总耗时"，而不必让一个 span 长时间开着。

### 5.3 is_tracing_available

[L148-157](../../vllm/tracing/__init__.py#L148)：检查任一后端可用。用来保护昂贵的 trace 逻辑：

```python
if is_tracing_available():
    headers = extract_trace_headers(raw_request.headers)
    # ... 构造属性、传播上下文
```

---

## 6. 属性常量：SpanAttributes

[utils.py:15](../../vllm/tracing/utils.py#L15) 定义了统一的 span 属性名，分两组：

### 6.1 GenAI 语义约定

对齐 OpenTelemetry GenAI semantic conventions：

- **用量**：`gen_ai.usage.prompt_tokens`、`gen_ai.usage.completion_tokens`、`gen_ai.usage.num_sequences`；
- **请求**：`gen_ai.request.max_tokens`、`gen_ai.request.temperature`、`gen_ai.request.top_p`、`gen_ai.request.n`、`gen_ai.request.id`；
- **响应**：`gen_ai.response.model`。

### 6.2 延迟分解（vLLM 自定义）

这些是性能分析的关键：

| 属性 | 含义 |
|------|------|
| `gen_ai.latency.time_in_queue` | 在调度器队列里等待的时间 |
| `gen_ai.latency.time_to_first_token` | TTFT |
| `gen_ai.latency.e2e` | 端到端 |
| `gen_ai.latency.time_in_scheduler` | 调度耗时 |
| `gen_ai.latency.time_in_model_forward` | 模型 forward 总耗时 |
| `gen_ai.latency.time_in_model_execute` | ModelRunner.execute 耗时 |
| `gen_ai.latency.time_in_model_prefill` | prefill 阶段 |
| `gen_ai.latency.time_in_model_decode` | decode 阶段 |
| `gen_ai.latency.time_in_model_inference` | 推理总时间 |

### 6.3 代码级属性

[LoadingSpanAttributes](../../vllm/tracing/utils.py#L48)：`code.namespace`、`code.function`、`code.filepath`、`code.lineno`，由 `@instrument` 自动填充。

---

## 7. Trace Header 工具

[utils.py](../../vllm/tracing/utils.py) 定义 W3C Trace Context 的两个标准 header：

```python
TRACE_HEADERS = ["traceparent", "tracestate"]
```

- `contains_trace_headers(headers)`：判断是否带追踪上下文；
- `extract_trace_headers(headers)`：只挑出这两个 header，用于透传给子进程/非 OTel 客户端；
- `log_tracing_disabled_warning()`：`@run_once`，收到带 trace header 的请求但 tracing 未启用时只警告一次。

`extract_trace_context(headers)` 在 otel.py 里，把 header 解析成 OTel `Context` 对象。

---

## 8. 在整个代码库的埋点分布

`@instrument` 当前装饰在 15 个文件的关键路径上，覆盖：

| 子系统 | 文件 | 埋点的阶段 |
|--------|------|-----------|
| 引擎 | `v1/engine/core.py` | EngineCore 单步、输入处理 |
| 引擎 | `v1/engine/core_client.py` | 跨进程客户端调用 |
| Worker | `v1/worker/gpu_worker.py`、`worker_base.py`、`cpu_model_runner.py`、`gpu_model_runner.py` | 模型执行 |
| Executor | `v1/executor/abstract.py`、`multiproc_executor.py` | 执行器与 collective_rpc |
| 模型加载 | `model_loader/base_loader.py`、`default_loader.py`、`weight_utils.py`、`utils.py` | 权重加载 |
| 编译 | `compilation/backends.py` | Dynamo/Inductor 编译阶段（含 `instrument_manual`） |
| 预热 | `warmup/deep_gemm_warmup.py` | kernel 预热 |
| API | `entrypoints/openai/api_server.py` | 请求入口 |

这些埋点在 OTel 后端（如 Jaeger/Tempo）里会形成一棵跨进程的 span 树，从 HTTP 请求一路到 GPU kernel。

---

## 小结

1. `vllm/tracing/` 是一个**极简门面**：三个文件、十几个公共 API，底层默认用 OpenTelemetry；
2. **`_REGISTERED_TRACING_BACKENDS` 注册表**让后端可插拔，所有门面函数都先检查 `is_available()`，未启用时零开销；
3. 生命周期分两类：`init_tracer` 在 API/主进程、`maybe_init_worker_tracer` 在 EngineCore/Worker 子进程；
4. 埋点用 `@instrument`（装饰器，自动包 span）或 `instrument_manual`（显式时间戳，记录排队/E2E 耗时）；
5. `SpanAttributes` 统一了 GenAI 语义约定与 vLLM 的延迟分解属性；
6. trace 上下文通过 W3C `traceparent`/`tracestate` header 与环境变量跨进程传播。

下一篇 [Tracing 之二：OpenTelemetry 后端实现](./tracing-02-otel后端实现.md) 将深入 `otel.py`，看 TracerProvider 如何建立、span 如何创建、同步/异步包装器以及子进程上下文传播的具体机制。
