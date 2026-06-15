# V1 引擎核心

> 源码路径: `vllm/v1/engine/`

V1 引擎是 vLLM 的默认执行引擎。顶层 `vllm.engine` 和 `vllm.entrypoints.llm` 均为 V1 实现的薄别名。本模块包含引擎的核心循环、前后端通信、输入/输出处理等关键组件。

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       Frontend (调用方)                          │
│  AsyncLLM / LLMEngine                                          │
└──────┬──────────────────────┬───────────────────────────────────┘
       │ add_request          │ get_output
       │ (EngineCoreRequest)  │ (EngineCoreOutputs)
       ▼                      ▲
┌─────────────────────────────────────────────────────────────────┐
│                  EngineCoreClient (通信层)                       │
│  InprocClient / SyncMPClient / AsyncMPClient                   │
│  ── ZMQ socket / in-process queue ──                            │
└──────┬──────────────────────┬───────────────────────────────────┘
       │                      │
       ▼                      │
┌─────────────────────────────────────────────────────────────────┐
│                    EngineCore (核心循环)                         │
│  ┌─────────────┐  ┌────────────┐  ┌─────────────────────┐      │
│  │ InputProc   │  │ Scheduler  │  │ ModelExecutor       │      │
│  │ (预处理)    │──│ (调度)     │──│ (模型执行)          │      │
│  └─────────────┘  └─────┬──────┘  └─────────────────────┘      │
│                         │                                       │
│                   SchedulerOutput                               │
│                         │                                       │
│                   ModelRunnerOutput                             │
│                         │                                       │
│                  OutputProcessor (后处理)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心文件一览

| 文件 | 职责 |
|------|------|
| `__init__.py` | 核心数据结构定义：`EngineCoreRequest`、`EngineCoreOutput`、`FinishReason` 等 |
| `core.py` | `EngineCore` — 引擎内循环；`EngineCoreProc` — 多进程封装；`DPEngineCoreProc` — 数据并行扩展 |
| `core_client.py` | `EngineCoreClient` — 前后端通信抽象（Inproc/MP/AsyncMP） |
| `async_llm.py` | `AsyncLLM` — 异步引擎前端（API Server 使用） |
| `llm_engine.py` | `LLMEngine` — 同步引擎前端（离线批量使用） |
| `input_processor.py` | `InputProcessor` — 输入验证、分词、构建 `EngineCoreRequest` |
| `output_processor.py` | `OutputProcessor` — 反分词、logprobs、构建 `RequestOutput` |
| `detokenizer.py` | `IncrementalDetokenizer` — 增量反分词与 stop string 检测 |
| `logprobs.py` | `LogprobsProcessor` — 对数概率处理 |
| `parallel_sampling.py` | `ParentRequest` — 并行采样（n>1）的子请求拆分与聚合 |
| `coordinator.py` | `DPCoordinator` — 数据并行协调器 |
| `tensor_ipc.py` | `TensorIpcSender/Receiver` — 跨进程 Tensor 共享（多模态） |
| `exceptions.py` | `EngineDeadError`、`EngineGenerateError` |
| `utils.py` | ZMQ 地址生成、进程管理、握手指引等辅助函数 |

---

## 核心数据结构 (`__init__.py`)

### FinishReason

请求结束原因的枚举，使用 `IntEnum` 以实现紧凑序列化：

| 值 | 含义 |
|----|------|
| `STOP (0)` | 遇到 stop string |
| `LENGTH (1)` | 达到 max_tokens 或 max_model_len |
| `ABORT (2)` | 客户端主动中止 |
| `ERROR (3)` | 可重试的内部错误（如 KV 加载失败） |
| `REPETITION (4)` | 检测到重复 token 模式（幻觉检测） |

### EngineCoreRequest

使用 `msgspec.Struct` 定义，`array_like=True` + `omit_defaults=True` + `gc=False` 以优化序列化性能和内存开销。关键字段：

- `request_id` — 内部唯一 ID（由 `InputProcessor.assign_request_id()` 在外部 ID 后追加随机后缀生成）
- `external_req_id` — 用户提供的原始 request ID
- `prompt_token_ids` — 输入 token ID 列表
- `prompt_embeds` — 输入 embedding（替代 token IDs 的场景）
- `prompt_is_token_ids` — 混合模式（token + embeds）的位置掩码
- `mm_features` — 多模态特征列表
- `sampling_params` / `pooling_params` — 采样或池化参数
- `lora_request` — LoRA 适配器请求
- `cache_salt` — 前缀缓存盐值
- `data_parallel_rank` — 数据并行目标 rank
- `client_index` — 输出回送目标客户端索引
- `current_wave` — DP 场景下请求所属的波次
- `priority` — 调度优先级
- `trace_headers` — 分布式追踪头
- `resumable` — 是否支持流式输入续传
- `reasoning_ended` / `reasoning_parser_kwargs` — 推理模式（如 o1）相关
- `abort_immediately` — 入队后立即中止（用于 KV transfer 清理）

### EngineCoreOutput

每个推理步中每个请求的输出：

- `new_token_ids` — 本步新生成的 token ID 列表
- `new_logprobs` / `new_prompt_logprobs_tensors` — logprobs
- `pooling_output` — 池化模型输出
- `finish_reason` / `stop_reason` — 结束原因
- `events` — 引擎核心事件（QUEUED / SCHEDULED / PREEMPTED）及其时间戳
- `kv_transfer_params` — KV 迁移参数
- `prefill_stats` — prefill 阶段统计
- `routed_experts` — MoE 路由专家信息
- `num_nans_in_logits` — logits 中 NaN 数（>0 表示输出损坏）

### EngineCoreOutputs

一个完整迭代步的输出集合：

- `engine_index` — 引擎索引（DP 场景）
- `outputs` — `EngineCoreOutput` 列表
- `scheduler_stats` — 调度器统计
- `utility_output` — 工具方法调用返回
- `finished_requests` — 已完成请求 ID 集合
- `wave_complete` / `start_wave` — DP 波次控制信号

### EngineCoreRequestType

请求类型枚举（以 hex byte 定义，可直接通过 socket 发送无需额外编码）：

| 值 | 类型 | 含义 |
|----|------|------|
| `\x00` | ADD | 添加新请求 |
| `\x01` | ABORT | 中止请求 |
| `\x02` | START_DP_WAVE | 启动 DP 新波次 |
| `\x03` | UTILITY | 工具方法调用 |
| `\x04` | EXECUTOR_FAILED | 执行器失败（内部哨兵） |
| `\x05` | WAKEUP | 唤醒关闭期间阻塞的 input_queue |

---

## EngineCore (`core.py`)

`EngineCore` 是 vLLM 引擎的**内循环**——调度、执行、输出三阶段的紧凑循环。

### 初始化流程

```python
EngineCore.__init__(vllm_config, executor_class, log_stats, ...)
```

1. **加载插件** — `load_general_plugins()`
2. **创建 ModelExecutor** — 负责模型加载与 GPU 执行
3. **初始化 KV Cache** — `_initialize_kv_caches()`:
   - 获取 KV cache 规格 (`model_executor.get_kv_cache_specs()`)
   - 内存探测确定可用 GPU 内存 (`model_executor.determine_available_memory()`)
   - 计算 KV cache 配置（含 auto-fit: 可能缩减 `max_model_len`）
   - 若 auto-fit 缩减了 max_model_len，同步更新到 workers
   - 初始化 KV cache 并执行 warmup (`model_executor.initialize_from_config()`)
4. **创建 StructuredOutputManager** — 结构化输出（JSON schema 等）管理
5. **创建 Scheduler** — `scheduler_config.get_scheduler_cls()` 工厂方法
6. **初始化 KV Connector**（若配置）— 跨引擎 KV 迁移
7. **初始化 Batch Queue**（流水线并行场景）— 异步调度与执行解耦
8. **初始化前缀缓存哈希器**（若启用前缀缓存或 KV connector）
9. **冻结 GC 堆** — `freeze_gc_heap()` 减少后续 GC 暂停
10. **启用环境变量缓存** — `enable_envs_cache()`

### 核心循环: `step()`

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    model_output = future.result()
    # 处理中止队列
    self._process_aborts_queue()
    # 更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

每一步的流程：
1. **Schedule** — 调度器决定哪些请求可以推进，分配 KV cache block
2. **Execute** — ModelExecutor 异步执行模型前向
3. **Sample** — 采样 token（含 grammar bitmask 约束）
4. **Process Aborts** — 处理模型执行期间到来的中止请求
5. **Update** — 将模型输出反馈给调度器，更新请求状态

### Batch Queue 模式 (`step_with_batch_queue`)

流水线并行场景下，`batch_queue_size > 1` 时启用。核心思路：
- 调度新 batch 的优先级高于获取已有 batch 的输出
- batch 满时才阻塞等待最早的 batch 完成
- 实现调度与执行的重叠，消除流水线气泡

### 暂停/恢复/休眠机制

**暂停调度 (`pause_scheduler`)**:
- `abort` 模式：立即中止所有请求
- `wait` 模式：等待在途请求完成（仅多进程模式）
- `keep` 模式：冻结队列中的请求，恢复后继续

**休眠 (`sleep`)**:
- Level 0：仅暂停调度，不改变 GPU 内存
- Level 1：卸载模型权重到 CPU，丢弃 KV cache
- Level 2：丢弃所有 GPU 内存

**唤醒 (`wake_up`)**:
- 恢复调度，按需重载权重

---

## EngineCoreProc (`core.py`)

`EngineCoreProc` 继承 `EngineCore`，是**多进程模式**下的封装，通过 ZMQ socket 与前端进程通信。

### 架构

```
Frontend Process                    EngineCore Process
┌──────────────────┐               ┌──────────────────────┐
│ EngineCoreClient │◄──ZMQ socket──►│ EngineCoreProc       │
│                  │               │ ┌──────────────────┐  │
│  input socket    │───request────►│ │ input_thread     │  │
│  output socket   │◄──outputs─────│ │ output_thread    │  │
│                  │               │ └──────────────────┘  │
│                  │               │ ┌──────────────────┐  │
│                  │               │ │ core_busy_loop   │  │
│                  │               │ └──────────────────┘  │
└──────────────────┘               └──────────────────────┘
```

### 关键设计

1. **双线程 IO** — `input_thread` 和 `output_thread` 分别处理 ZMQ socket 的收发，释放 GIL 以实现与 GPU 执行的 IO 重叠
2. **msgpack 序列化** — 使用 `msgspec` 实现高性能编解码，`array_like=True` 减少内存分配
3. **Tensor IPC** — 多模态场景下通过 `torch.multiprocessing.Queue` 共享 GPU tensor，避免序列化
4. **启动握手** — `_perform_handshakes()` 通过 ZMQ DEALER/ROUTER 模式交换地址信息
5. **DP Coordinator 集成** — 支持 DP>1 时与 `DPCoordinator` 协调

### 忙循环 (`run_busy_loop`)

```python
def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()   # 处理输入直到有工作要做
        self._process_engine_step()   # 执行一步引擎循环
```

- `_process_input_queue`: 阻塞等待直到调度器有请求或有新输入到达
- `_process_engine_step`: 执行 `step()` 并将输出放入 `output_queue`
- 空闲时通知 `_idle_state_callbacks`（用于暂停等待）

### 优雅关闭

- 收到 SIGTERM/SIGINT → 设置 `shutdown_state = REQUESTED`
- 若 `shutdown_timeout = 0`：立即中止所有请求
- 若 `shutdown_timeout > 0`：排空在途请求后关闭
- 通过 `ENGINE_CORE_DEAD` 消息通知前端

---

## DPEngineCoreProc (`core.py`)

数据并行（MoE 模型专用）的引擎核心进程扩展。

### 核心机制

1. **波次调度 (Wave-based scheduling)** — 请求按"波次"分发到各 DP rank，同一波次内的请求在各 rank 上并行执行
2. **全局未完成检测** — 每 32 步通过 all-reduce 同步各 rank 的未完成请求状态
3. **两阶段暂停协议** — 暂停需所有 DP rank 达成共识，通过 all-reduce 实现
4. **DP 负载均衡** — 向 Coordinator 发布请求计数，支持内部/混合负载均衡模式

### 弹性专家并行 (Elastic EP)

支持运行时动态扩缩 DP size：
- `reinitialize_distributed()` — 接收重配置请求，创建 `ElasticEPScalingState`
- 扩容时新引擎通过 `EEPNotificationType` 通知已有引擎
- 缩容时被移除的引擎优雅退出

---

## EngineCoreClient (`core_client.py`)

前后端通信抽象基类，三种实现：

| 客户端 | 场景 | 通信方式 |
|--------|------|----------|
| `InprocClient` | 调试/单进程模式 | 直接调用 EngineCore 方法 |
| `SyncMPClient` | LLMEngine (同步离线) | ZMQ socket + 后台 EngineCoreProc |
| `AsyncMPClient` | AsyncLLM (异步服务) | ZMQ asyncio socket + 后台 EngineCoreProc |

### 关键方法

- `add_request(request)` / `add_request_async(request)` — 添加请求
- `get_output()` / `get_output_async()` — 获取输出
- `abort_requests(request_ids)` / `abort_requests_async(request_ids)` — 中止请求
- `shutdown(timeout)` — 关闭引擎核心进程

---

## AsyncLLM (`async_llm.py`)

异步引擎前端，是 API Server 的主要入口。实现 `EngineClient` 协议。

### 请求处理流程

```
API Server
    │
    │ generate(prompt, sampling_params, request_id)
    ▼
AsyncLLM.add_request()
    │
    ├── InputProcessor.process_inputs()  ← 验证 + 分词 + 构建 EngineCoreRequest
    ├── InputProcessor.assign_request_id()  ← 生成内部唯一 ID
    ├── OutputProcessor.add_request()  ← 注册请求状态
    ├── EngineCoreClient.add_request_async()  ← 发送到后台进程
    │
    │ 返回 RequestOutputCollector
    ▼
AsyncLLM.generate() 循环
    │
    ├── 从 RequestOutputCollector.get() 取输出
    └── yield RequestOutput
```

### 输出处理器 (`_run_output_handler`)

后台 asyncio task 持续运行：
1. 从 `EngineCoreClient.get_output_async()` 拉取输出
2. 分块处理（`VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`），避免阻塞事件循环
3. 调用 `OutputProcessor.process_outputs()` 进行反分词和后处理
4. 处理完毕的 `RequestOutput` 推入各请求的 `RequestOutputCollector`
5. 中止因 stop string 完成的请求

### 流式输入

支持通过 `AsyncGenerator[StreamingInput, None]` 逐步追加输入：
- 每个输入 chunk 创建独立的 `EngineCoreRequest`（`resumable=True`）
- 输出处理器在子请求完成后应用挂起的更新
- 最终发送 `resumable=False` 的请求标记输入结束

### 并行采样 (n>1)

当 `SamplingParams.n > 1` 时：
1. 创建 `ParentRequest` 管理子请求
2. 为每个 n 创建独立的子请求（各自 `n=1`，不同 seed）
3. 子请求输出由 `ParentRequest.get_outputs()` 聚合
4. 所有子请求完成后才标记为 finished

---

## LLMEngine (`llm_engine.py`)

同步引擎前端，用于离线批量推理。与 `AsyncLLM` 共享 `InputProcessor` 和 `OutputProcessor`，但使用同步的 `EngineCoreClient`。

### 核心方法

- `add_request()` — 添加请求（同步）
- `step()` — 执行一步引擎循环，返回 `RequestOutput` 列表
- `has_unfinished_requests()` — 检查是否还有未完成请求

### step() 流程

1. 从 `EngineCoreClient.get_output()` 获取输出
2. `OutputProcessor.process_outputs()` 后处理
3. 中止因 stop string 完成的请求
4. 记录统计信息

---

## InputProcessor (`input_processor.py`)

将用户输入转换为 `EngineCoreRequest`。

### 处理流程

1. **参数验证** — `_validate_params()`: 检查采样/池化参数与支持的任务类型
2. **LoRA 验证** — `_validate_lora()`: 检查 LoRA 是否已启用
3. **输入预处理** — 调用 `InputPreprocessor.preprocess()` 进行分词和多模态处理
4. **平台验证** — `current_platform.validate_request()`
5. **编码器-解码器拆分** — `split_enc_dec_input()` 分离 encoder/decoder 输入
6. **长度验证** — `_validate_prompt_len()`: 检查不超过 max_model_len
7. **参数补全** — 克隆 SamplingParams，设置默认 max_tokens，应用 generation_config
8. **多模态特征排序** — 按 position 排序，合并为 `MultiModalFeatureSpec` 列表
9. **构建 EngineCoreRequest**

### 请求 ID 机制

`assign_request_id()` 在用户提供的 request_id 后追加 8 位随机字符，确保内部唯一性。原始 ID 保存在 `external_req_id` 字段。可通过 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` 禁用。

---

## OutputProcessor (`output_processor.py`)

将 `EngineCoreOutput` 转换为面向用户的 `RequestOutput`。

### 核心数据结构

**RequestState** — 每个请求的完整状态：
- 反分词器 (`IncrementalDetokenizer`)
- Logprobs 处理器 (`LogprobsProcessor`)
- 流式控制（stream_interval, output_kind）
- 统计信息 (`RequestStateStats`)
- 流式输入队列 (`input_chunk_queue`)

**RequestOutputCollector** — 异步场景下的输出收集器：
- 支持 DELTA / CUMULATIVE / FINAL_ONLY 三种输出模式
- DELTA 模式下自动合并连续输出

### process_outputs() 流程

这是 V1 中**唯一**遍历完整 batch 的循环，所有需要逐请求处理的逻辑都应在此处：

1. **统计更新** — 计算每请求的迭代统计
2. **反分词** — `detokenizer.update()`，检测 stop string
3. **Logprobs 处理** — `logprobs_processor.update_from_output()`
4. **构建 RequestOutput** — 根据输出模式（DELTA/CUMULATIVE/FINAL_ONLY）和 stream_interval 决定是否输出
5. **完成处理** — 清理请求状态，若因 stop string 结束但 EngineCore 未标记完成则追加中止

---

## IncrementalDetokenizer (`detokenizer.py`)

增量反分词器，两种实现：

| 实现 | 条件 | 特点 |
|------|------|------|
| `FastIncrementalDetokenizer` | tokenizers >= 0.22.0 + PreTrainedTokenizerFast | 使用 `DecodeStream` 原生 prefill |
| `SlowIncrementalDetokenizer` | 其他 | Python 实现的增量反分词 |

### 核心功能

- **增量反分词** — 每步仅解码新 token，维护内部偏移状态
- **Stop string 检测** — `check_stop_strings()`: 在输出文本中搜索 stop string
- **缓冲机制** — 当 `include_stop_str_in_output=False` 时，持有最多 `max(stop_str_len) - 1` 个字符不输出，确保 stop string 被截断
- **min_tokens 支持** — 在达到 min_tokens 之前跳过 stop string 检测

---

## ParentRequest (`parallel_sampling.py`)

并行采样（`n>1`）的管理器。

### 工作方式

1. 将父请求拆分为 n 个子请求，每个 `n=1`
2. 子请求 ID 格式：`{index}_{parent_request_id}`
3. 若有 seed，每个子请求分配唯一 seed (`seed + index`)
4. 输出聚合：
   - FINAL_ONLY 模式：等待所有子请求完成后一次性返回
   - 其他模式：子请求完成即返回
5. 统计：追踪最大生成 token 数，用于迭代统计

---

## DPCoordinator (`coordinator.py`)

数据并行协调器，在 DP>1 且使用内部负载均衡时运行在前端进程中。

### 职责

- 管理各 DP rank 的请求分配
- 收集各 rank 的请求计数进行负载均衡
- 协调 DP 波次的启动与完成
- 通过 ZMQ XSUB/XPUB 与各 EngineCoreProc 通信

---

## TensorIPC (`tensor_ipc.py`)

跨进程 Tensor 共享机制，用于多模态场景。

### 工作原理

- `TensorIpcSender` (前端): 将 tensor 移至共享内存，通过 `torch.multiprocessing.Queue` 发送
- `TensorIpcReceiver` (EngineCore): 从队列接收，按 (sender_id, message_id, tensor_id) 索引
- 接收端采用 drain-and-buffer 模式，容忍乱序到达

---

## 关键交互流程

### 完整请求生命周期

```
1. 客户端调用 generate(prompt, params)
2. InputProcessor.process_inputs() → EngineCoreRequest
3. OutputProcessor.add_request() → 创建 RequestState
4. EngineCoreClient.add_request_async() → 发送到后台进程
5. EngineCoreProc.input_thread 接收 → 放入 input_queue
6. core_busy_loop 取出 → EngineCore.add_request()
7. Scheduler.add_request() → 加入等待队列
8. [调度] Scheduler.schedule() → SchedulerOutput
9. [执行] ModelExecutor.execute_model() → ModelRunnerOutput
10. [采样] ModelExecutor.sample_tokens() → token_ids
11. Scheduler.update_from_output() → EngineCoreOutputs
12. output_thread 发送回前端
13. EngineCoreClient.get_output_async() 接收
14. OutputProcessor.process_outputs():
    a. IncrementalDetokenizer.update() → 反分词 + stop 检测
    b. LogprobsProcessor.update_from_output()
    c. RequestState.make_request_output() → RequestOutput
15. RequestOutputCollector.put() → 推入队列
16. generate() 循环 yield RequestOutput
17. finish_reason != None → 清理 RequestState
```

---

## 与其他模块的关系

| 模块 | 交互方式 |
|------|----------|
| `vllm/v1/core/` (Scheduler) | EngineCore 每步调用 `scheduler.schedule()` 和 `scheduler.update_from_output()` |
| `vllm/v1/worker/` (Worker) | 通过 ModelExecutor 间接调用，执行模型前向 |
| `vllm/v1/executor/` | EngineCore 持有 ModelExecutor 实例，负责模型执行、KV cache 初始化 |
| `vllm/v1/attention/` | 通过 Scheduler 和 ModelExecutor 间接使用 |
| `vllm/v1/structured_output/` | EngineCore 持有 StructuredOutputManager，管理 grammar 初始化 |
| `vllm/config/` | 所有组件通过 `VllmConfig` 共享配置 |
| `vllm/renderers/` | InputProcessor 和 AsyncLLM 使用 Renderer 进行输入渲染 |
| `vllm/lora/` | EngineCore 代理 LoRA 操作到 ModelExecutor |
