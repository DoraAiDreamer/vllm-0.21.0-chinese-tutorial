# vLLM 日志系统详解

> 基于 vllm-0.21.0 版本

本文档全面介绍 vLLM 的日志系统，包括日志格式、自定义输出目录、日志内容控制、KV Cache 用量监控、性能指标、结构化指标（Prometheus/OpenTelemetry）等。

---

## 一、日志系统架构

### 1.1 整体结构

vLLM 使用 Python 标准 `logging` 模块，通过自定义配置实现结构化日志输出。

```
┌────────────────────────────────────────────────────────────┐
│                     日志调用方                              │
│  vllm/v1/engine/ | vllm/v1/core/ | vllm/entrypoints/ ...  │
│       │                                                    │
│       ▼ init_logger(__name__)                              │
│  logging.getLogger(name) + 注入 debug_once/info_once/      │
│  warning_once 方法                                         │
│       │                                                    │
│       ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  vllm 根 logger                                     │   │
│  │  - handler: StreamHandler (stdout/stderr)           │   │
│  │  - formatter: ColoredFormatter / NewLineFormatter   │   │
│  │  - level: INFO (默认)                               │   │
│  └─────────────────────────────────────────────────────┘   │
│       │                                                    │
│       ▼                                                    │
│  输出到控制台或文件                                        │
└────────────────────────────────────────────────────────────┘
```

### 1.2 日志格式

默认日志格式：

```
[YYYY-MM-DD HH:MM:SS] LEVEL [file.py:line] message
```

实际输出示例：

```
[06-15 14:32:01] INFO [model_executor/layers/quantization/fp8.py:42] Loaded FP8 model
[06-15 14:32:03] WARNING [v1/engine/core.py:110] KV cache usage: 45.2%
[06-15 14:32:05] ERROR [distributed/parallel_state.py:234] NCCL init failed
```

**颜色输出**（终端下自动启用）：
| 级别 | 颜色 |
|------|------|
| DEBUG | 白色 |
| INFO | 绿色 |
| WARNING | 黄色 |
| ERROR | 红色 |
| CRITICAL | 品红 |

### 1.3 路径缩写

DEBUG 级别下，文件路径会被缩写：
- `vllm/model_executor/layers/quantization/utils/fp8_utils.py` → `model_executor/.../quantization/utils/fp8_utils.py`
- `vllm/v1/engine/core.py` → `v1/engine/core.py`

---

## 二、环境变量配置

### 2.1 日志级别控制

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `VLLM_LOGGING_LEVEL` | `INFO` | 日志级别：`DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `VLLM_LOGGING_PREFIX` | `""` | 所有日志消息的前缀 |
| `VLLM_LOGGING_STREAM` | `ext://sys.stdout` | 日志输出流 |
| `VLLM_LOGGING_COLOR` | `auto` | 颜色输出：`auto` / `1`（开）/ `0`（关） |
| `VLLM_CONFIGURE_LOGGING` | `True` | 是否使用 vLLM 默认日志配置 |
| `VLLM_LOGGING_CONFIG_PATH` | `None` | 自定义 JSON 日志配置文件路径 |

### 2.2 日志级别设置

```bash
# 设置为 DEBUG（最详细）
export VLLM_LOGGING_LEVEL=DEBUG

# 设置为 WARNING（仅警告和错误）
export VLLM_LOGGING_LEVEL=WARNING

# 设置为 ERROR（仅错误）
export VLLM_LOGGING_LEVEL=ERROR
```

```python
# Python 代码中设置
import os
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

# 必须在导入 vllm 之前设置
from vllm import LLM
```

### 2.3 日志前缀

```bash
# 所有日志添加前缀
export VLLM_LOGGING_PREFIX="[my-service] "
```

输出示例：
```
[my-service] [06-15 14:32:01] INFO [model.py:42] Model loaded
```

### 2.4 颜色控制

```bash
# 强制启用颜色
export VLLM_LOGGING_COLOR=1

# 强制禁用颜色（适合写入文件时）
export VLLM_LOGGING_COLOR=0

# NO_COLOR 环境变量（标准做法）
export NO_COLOR=1
```

---

## 三、自定义日志输出目录和文件

### 3.1 方法一：Shell 重定向（最简单）

```bash
# 输出到文件
vllm serve Qwen/Qwen2.5-7B-Instruct > vllm.log 2>&1

# 追加模式
vllm serve Qwen/Qwen2.5-7B-Instruct >> vllm.log 2>&1

# 使用 tee 实时查看+保存
vllm serve Qwen/Qwen2.5-7B-Instruct 2>&1 | tee vllm.log
```

### 3.2 方法二：自定义日志配置文件（推荐）

vLLM 支持通过 JSON 配置文件完全自定义日志行为：

```json
// vllm_logging.json
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "()": "vllm.logging_utils.NewLineFormatter",
            "format": "%(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "INFO",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "level": "DEBUG",
            "filename": "/path/to/logs/vllm_debug.log",
            "encoding": "utf-8"
        },
        "error_file": {
            "class": "logging.FileHandler",
            "formatter": "detailed",
            "level": "ERROR",
            "filename": "/path/to/logs/vllm_error.log",
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "vllm": {
            "handlers": ["console", "file", "error_file"],
            "level": "DEBUG",
            "propagate": false
        }
    }
}
```

```bash
# 使用自定义配置文件
export VLLM_CONFIGURE_LOGGING=True
export VLLM_LOGGING_CONFIG_PATH=/path/to/vllm_logging.json
vllm serve Qwen/Qwen2.5-7B-Instruct
```

### 3.3 方法三：Python 代码自定义

```python
import os
import logging
import logging.config

# 在导入 vllm 之前配置
LOG_DIR = "/path/to/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "level": "DEBUG",
            "filename": os.path.join(LOG_DIR, "vllm.log"),
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "vllm": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}

# 设置环境变量
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["VLLM_CONFIGURE_LOGGING"] = "False"  # 禁用默认配置

# 应用配置
logging.config.dictConfig(logging_config)

# 然后导入 vllm
from vllm import LLM
```

### 3.4 方法四：按级别分离日志文件

```python
import os
import logging
import logging.config

LOG_DIR = "/path/to/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 创建 vllm 的 root logger handler
root_logger = logging.getLogger("vllm")

# INFO 级别 → 主日志文件
info_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "vllm_info.log"),
    maxBytes=50 * 1024 * 1024,
    backupCount=3,
)
info_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(fileinfo)s:%(lineno)d] %(message)s"
))
info_handler.setLevel(logging.INFO)
root_logger.addHandler(info_handler)

# ERROR 级别 → 错误日志文件
error_handler = logging.handlers.RotatingFileHandler(
    os.path.join(LOG_DIR, "vllm_error.log"),
    maxBytes=50 * 1024 * 1024,
    backupCount=3,
)
error_handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(fileinfo)s:%(lineno)d] %(message)s"
))
error_handler.setLevel(logging.ERROR)
root_logger.addHandler(error_handler)

# 控制台保持默认
root_logger.setLevel(logging.INFO)

os.environ["VLLM_CONFIGURE_LOGGING"] = "False"

from vllm import LLM
```

---

## 四、日志内容控制

### 4.1 引擎统计日志（log_stats）

**控制参数：** `disable_log_stats`

**默认值：** `True`（API Server 模式默认禁用，离线推理默认启用）

**作用：** 控制调度器是否定期输出统计信息，包括：
- 运行中请求数
- 排队中请求数
- KV Cache 使用率
- 前缀缓存命中率
- 推测解码统计
- CUDA Graph 统计
- 性能指标（tokens/sec 等）

#### 4.1.1 启用引擎统计日志

```python
from vllm import LLM

# 离线推理时默认已启用
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# API Server 模式下显式启用
vllm serve Qwen/Qwen2.5-7B-Instruct --disable-log-stats=False
```

```bash
# CLI 方式
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --disable-log-stats=False \
    --log-stats-interval 10  # 每 10 秒输出一次
```

#### 4.1.2 日志输出间隔

**环境变量：** `VLLM_LOG_STATS_INTERVAL`（秒，默认 10.0）

```bash
# 每 5 秒输出一次统计
export VLLM_LOG_STATS_INTERVAL=5

# 禁用定期统计（设为负数）
export VLLM_LOG_STATS_INTERVAL=-1
```

#### 4.1.3 典型统计日志输出

```
[06-15 14:32:01] INFO [v1/core/sched/scheduler.py:XXX] 
  - avg_prompt_throughput: 1234.5 tokens/s
  - avg_generation_throughput: 567.8 tokens/s
  - running: 8, waiting: 3, preempted: 0
  - KV cache usage: 45.2%
  - Prefix cache hit rate: 67.3%
  - CUDA Graph: 128/256 captured
```

### 4.2 请求日志（log_requests）

**控制参数：** `disable_log_requests`

**作用：** 控制是否记录每个请求的详细信息（输入 prompt、输出文本等）。

```python
# 禁用请求日志（生产环境推荐，减少日志量）
llm = LLM(model="model", disable_log_requests=True)

# 启用请求日志（调试用）
vllm serve MODEL --disable-log-requests=False
```

### 4.3 请求批次大小日志

**环境变量：** `VLLM_LOG_BATCHSIZE_INTERVAL`（秒）

```bash
# 每 30 秒输出批次大小统计
export VLLM_LOG_BATCHSIZE_INTERVAL=30

# 禁用（默认 -1）
export VLLM_LOG_BATCHSIZE_INTERVAL=-1
```

### 4.4 模型检查日志

**环境变量：** `VLLM_LOG_MODEL_INSPECTION`

```bash
# 启用模型检查日志（加载时输出模型结构信息）
export VLLM_LOG_MODEL_INSPECTION=1
```

---

## 五、KV Cache 用量监控

### 5.1 调度器统计中的 KV Cache 用量

当 `log_stats=True` 时，调度器每 `VLLM_LOG_STATS_INTERVAL` 秒输出一次统计，包含：

| 指标 | 说明 |
|------|------|
| `kv_cache_usage` | KV Cache 使用率（0.0~1.0） |
| `num_running_reqs` | 正在运行的请求数 |
| `num_waiting_reqs` | 等待中的请求数 |
| `prefix_cache_stats` | 前缀缓存统计（命中数、未命中数、缓存块数） |
| `cudagraph_stats` | CUDA Graph 统计 |
| `spec_decoding_stats` | 推测解码统计 |
| `perf_stats` | 性能统计（tokens/sec） |

### 5.2 KV Cache 指标（Prometheus）

vLLM 0.21.0 支持通过 `ObservabilityConfig` 收集 KV Cache 指标：

```python
from vllm.config import ObservabilityConfig

llm = LLM(
    model="model",
    observability_config=ObservabilityConfig(
        kv_cache_metrics=True,           # 启用 KV Cache 指标
        kv_cache_metrics_sample=0.01,    # 采样率 1%
        cudagraph_metrics=True,          # 启用 CUDA Graph 指标
        enable_mfu_metrics=True,         # 启用 MFU 指标
    ),
)
```

**CLI 方式（API Server）：**
```bash
vllm serve MODEL \
    --kv-cache-metrics \
    --kv-cache-metrics-sample 0.01 \
    --cudagraph-metrics \
    --enable-mfu-metrics
```

**KV Cache 指标内容：**
- **KV Cache 驻留时间**：每个 KV block 在缓存中的时间
- **空闲时间**：KV block 未被使用的时间
- **复用间隙**：KV block 被复用时跳过的 token 数
- **CUDA Graph 分发模式**：runtime cudagraph dispatch 模式及频率

### 5.3 KV Cache 事件发布

vLLM 支持通过 `EventPublisherFactory` 发布 KV Cache 事件（驱逐、加载等）：

```python
# 通过环境变量配置事件发布
export VLLM_KV_EVENTS_CONFIG='{"type": "prometheus", "endpoint": "http://localhost:9090"}'
```

### 5.4 KV Transfer 统计

当使用 KV Transfer（跨引擎 KV 迁移）时，统计中包含：

| 指标 | 说明 |
|------|------|
| `kv_connector_stats` | KV 连接器传输统计 |
| `connector_prefix_cache_stats` | 连接器前缀缓存统计 |
| `kv_cache_eviction_events` | KV Cache 驱逐事件 |

---

## 六、性能指标

### 6.1 性能统计（PerfStats）

调度器每步收集的性能数据：

```python
# 启用详细迭代日志
llm = LLM(
    model="model",
    observability_config=ObservabilityConfig(
        enable_logging_iteration_details=True,
    ),
)
```

**输出内容：**
- 上下文请求数 / 生成请求数
- 上下文 token 数 / 生成 token 数
- 每步 CPU 耗时

### 6.2 Model FLOPs Utilization (MFU)

```bash
vllm serve MODEL --enable-mfu-metrics
```

输出示例：
```
[06-15 14:32:01] INFO [v1/core/sched/scheduler.py:XXX]
  MFU: 45.2% (theoretical peak: 100%)
  Actual FLOPS: 123.4 TFLOPS
  Theoretical FLOPS: 273.0 TFLOPS
```

### 6.3 CUDA Graph 指标

```bash
vllm serve MODEL --cudagraph-metrics
```

输出内容：
- 填充 token 数 / 未填充 token 数
- CUDA Graph 分发模式
- 各模式频率

### 6.4 推测解码统计

当启用推测解码时，统计中包含：

| 指标 | 说明 |
|------|------|
| `num_draft_tokens` | 草稿 token 数 |
| `num_accepted_tokens` | 接受的 token 数 |
| `acceptance_rate` | 接受率 |
| `speedup` | 加速比 |

---

## 七、结构化指标（Prometheus / OpenTelemetry）

### 7.1 Prometheus 指标

vLLM 的 API Server 暴露 Prometheus 指标端点：

```bash
vllm serve MODEL --metrics
```

访问 `http://localhost:8000/metrics` 获取 Prometheus 格式指标。

**可用指标包括：**
- `vllm:prompt_tokens_total` — 总 prompt token 数
- `vllm:generation_tokens_total` — 总生成 token 数
- `vllm:time_to_first_token_seconds` — 首 token 延迟
- `vllm:time_per_output_token_seconds` — 每 token 延迟
- `vllm:e2e_request_latency_seconds` — 端到端请求延迟
- `vllm:request_success_total` — 成功请求数
- `vllm:gpu_cache_usage_perc` — GPU KV Cache 使用百分比
- `vllm:cpu_cache_usage_perc` — CPU KV Cache 使用百分比

### 7.2 OpenTelemetry 追踪

```python
from vllm.config import ObservabilityConfig

llm = LLM(
    model="model",
    observability_config=ObservabilityConfig(
        otlp_traces_endpoint="http://localhost:4317",
        collect_detailed_traces=["model", "worker"],  # 或 ["all"]
    ),
)
```

**CLI 方式：**
```bash
vllm serve MODEL \
    --otlp-traces-endpoint http://localhost:4317 \
    --collect-detailed-traces model,worker
```

**`collect_detailed_traces` 可选值：**
- `["model"]` — 收集模型前向传播时间
- `["worker"]` — 收集 worker 执行时间
- `["all"]` — 收集所有模块时间

**注意：** 详细追踪有性能开销，仅用于调试。

### 7.3 NVTX 层追踪

```python
llm = LLM(
    model="model",
    observability_config=ObservabilityConfig(
        enable_layerwise_nvtx_tracing=True,  # 逐层 NVTX 追踪
    ),
)
```

**注意：** 启用后 CUDA Graph 不可用。

---

## 八、API Server 访问日志

### 8.1 访问日志过滤

vLLM 的 API Server 使用 Uvicorn 的访问日志，支持排除特定端点：

```python
from vllm.logging_utils import create_uvicorn_log_config

# 排除 /health 和 /metrics 的访问日志
config = create_uvicorn_log_config(excluded_paths=["/health", "/metrics"])
```

### 8.2 访问日志格式

```
127.0.0.1:12345 - "GET /v1/models HTTP/1.1" 200
127.0.0.1:12346 - "POST /v1/chat/completions HTTP/1.1" 200
```

---

## 九、调试功能

### 9.1 函数调用追踪

vLLM 提供函数调用追踪功能，用于调试 hang 或崩溃：

```python
from vllm.logger import enable_trace_function_call

# 追踪 vllm 目录下所有函数调用
enable_trace_function_call("/path/to/trace.log")
```

输出示例：
```
2026-06-15 14:32:01.123456 Call to generate in llm.py:150 from __init__ in llm.py:100
2026-06-15 14:32:01.234567 Return from generate in llm.py:150 from __init__ in llm.py:100
```

**注意：** 这会显著降低性能，仅用于调试。

### 9.2 一次性日志

vLLM 提供 `info_once`、`debug_once`、`warning_once` 方法，确保相同消息只输出一次：

```python
logger = init_logger(__name__)

# 只在首次调用时输出
logger.info_once("Model loaded with %d layers", num_layers)

# 支持分布式作用域
logger.info_once("Engine initialized", scope="global")  # 仅全局首个 rank 输出
logger.info_once("Engine initialized", scope="local")   # 仅本地首个 rank 输出
```

### 9.3 日志抑制

```python
from vllm.logger import suppress_logging

# 临时抑制 INFO 及以上的日志
with suppress_logging(logging.INFO):
    # 此处的日志不会被输出
    some_noisy_operation()
```

---

## 十、不同运行模式的日志差异

### 10.1 离线推理（LLM 类）

```python
from vllm import LLM

llm = LLM(
    model="model",
    # disable_log_stats=True  (默认，但可通过 VLLM_LOG_STATS_INTERVAL 控制)
    # disable_log_requests=False (默认)
)
```

**输出内容：**
- 模型加载信息
- KV Cache 初始化信息
- 引擎配置摘要
- 定期统计（如果启用）

### 10.2 API Server（vllm serve）

```bash
vllm serve MODEL \
    --disable-log-stats=True \    # 默认禁用
    --disable-log-requests=True \ # 默认禁用
    --log-stats-interval 10
```

**输出内容：**
- Uvicorn 访问日志
- vLLM 引擎日志
- Prometheus 指标（如果启用）

### 10.3 CLI 命令

```bash
vllm chat MODEL
vllm complete MODEL --prompt "..."
```

**输出内容：**
- 简化的引擎日志
- 用户输出

---

## 十一、完整配置示例

### 11.1 生产环境配置

```bash
# 生产环境：仅 INFO 级别，输出到文件，禁用请求日志
export VLLM_LOGGING_LEVEL=INFO
export VLLM_LOGGING_COLOR=0
export VLLM_LOG_STATS_INTERVAL=30
export VLLM_LOG_BATCHSIZE_INTERVAL=-1
export VLLM_LOG_MODEL_INSPECTION=0

# 启动
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --disable-log-stats=True \
    --disable-log-requests=True \
    --log-stats-interval 30
```

### 11.2 开发调试配置

```bash
# 开发环境：DEBUG 级别，详细日志
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOG_STATS_INTERVAL=5
export VLLM_LOG_BATCHSIZE_INTERVAL=30
export VLLM_LOG_MODEL_INSPECTION=1

# 启动
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --disable-log-stats=False \
    --disable-log-requests=False
```

### 11.3 监控配置

```bash
# 启用 Prometheus 指标
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --kv-cache-metrics \
    --kv-cache-metrics-sample 0.01 \
    --cudagraph-metrics \
    --enable-mfu-metrics
```

### 11.4 完整 Python 配置

```python
import os
import logging
import logging.config

# ========== 日志配置 ==========
LOG_DIR = "/var/log/vllm"
os.makedirs(LOG_DIR, exist_ok=True)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(levelname)s] %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s",
            "datefmt": "%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout",
        },
        "info_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": "INFO",
            "filename": os.path.join(LOG_DIR, "vllm_info.log"),
            "maxBytes": 100 * 1024 * 1024,
            "backupCount": 10,
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "standard",
            "level": "ERROR",
            "filename": os.path.join(LOG_DIR, "vllm_error.log"),
            "maxBytes": 50 * 1024 * 1024,
            "backupCount": 10,
        },
    },
    "loggers": {
        "vllm": {
            "handlers": ["console", "info_file", "error_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
os.environ["VLLM_CONFIGURE_LOGGING"] = "False"
os.environ["VLLM_LOGGING_COLOR"] = "0"

logging.config.dictConfig(logging_config)

# ========== 引擎配置 ==========
from vllm import LLM, SamplingParams
from vllm.config import ObservabilityConfig

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    observability_config=ObservabilityConfig(
        kv_cache_metrics=True,
        kv_cache_metrics_sample=0.01,
        enable_mfu_metrics=True,
    ),
)

# ========== 推理 ==========
outputs = llm.generate(
    "Hello, world!",
    SamplingParams(temperature=0.7, max_tokens=128),
)
```

---

## 十二、日志排查常见问题

### 12.1 日志不输出

**排查步骤：**

```python
# 1. 检查 logger 级别
import logging
print(logging.getLogger("vllm").level)  # 应为 20 (INFO) 或更低

# 2. 检查 handler
print(logging.getLogger("vllm").handlers)

# 3. 检查环境变量
import os
print(os.environ.get("VLLM_LOGGING_LEVEL"))
print(os.environ.get("VLLM_CONFIGURE_LOGGING"))
```

### 12.2 日志重复输出

**原因：** 多个 handler 或 propagate=True。

**解决：**
```python
logger = logging.getLogger("vllm")
logger.propagate = False  # 禁止向上传播
# 确保只有一个 handler
logger.handlers.clear()
```

### 12.3 日志中缺少文件信息

**原因：** 非 DEBUG 级别下，文件路径显示为完整绝对路径。

**解决：** 设置 `VLLM_LOGGING_LEVEL=DEBUG` 启用路径缩写。

### 12.4 日志输出到 stderr 而非 stdout

**原因：** 某些 handler 默认使用 `ext://sys.stderr`。

**解决：**
```bash
# 指定输出到 stdout
export VLLM_LOGGING_STREAM=ext://sys.stdout

# 或指定文件
export VLLM_LOGGING_STREAM=ext://sys.stderr
```

---

## 十三、环境变量速查表

| 环境变量 | 默认值 | 作用 |
|----------|--------|------|
| `VLLM_LOGGING_LEVEL` | `INFO` | 日志级别 |
| `VLLM_LOGGING_PREFIX` | `""` | 日志前缀 |
| `VLLM_LOGGING_STREAM` | `ext://sys.stdout` | 输出流 |
| `VLLM_LOGGING_COLOR` | `auto` | 颜色输出 |
| `VLLM_CONFIGURE_LOGGING` | `True` | 是否使用默认配置 |
| `VLLM_LOGGING_CONFIG_PATH` | `None` | 自定义配置文件路径 |
| `VLLM_LOG_STATS_INTERVAL` | `10.0` | 统计日志间隔（秒） |
| `VLLM_LOG_BATCHSIZE_INTERVAL` | `-1` | 批次大小日志间隔（秒） |
| `VLLM_LOG_MODEL_INSPECTION` | `0` | 是否输出模型检查日志 |
| `NO_COLOR` | `""` | 禁用颜色（标准环境变量） |
