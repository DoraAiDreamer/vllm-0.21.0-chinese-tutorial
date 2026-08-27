# 调度与 KV Cache

> 源码路径: `vllm/v1/core/`

调度器（Scheduler）与 KV Cache 管理器是 vLLM 高吞吐推理的两大引擎。调度器决定"每一步让哪些请求、各推进多少 token"，KV Cache 管理器决定"这些 token 的键值缓存放在哪些物理块里、哪些历史块可以复用"。二者在 `schedule()` 中紧密协作，构成了连续批处理（continuous batching）、分块预填充（chunked prefill）、前缀缓存（prefix caching）、投机解码（speculative decoding）和滑动窗口（sliding window）等特性的实现基础。

本模块的设计哲学可以用 [scheduler.py:311-320](../../vllm/v1/core/sched/scheduler.py#L311-L320) 的一段注释概括：

> 调度器里没有独立的"decode 阶段"或"prefill 阶段"。每个请求只有 `num_computed_tokens` 和 `num_tokens_with_spec`。每一步，调度器都试图让每个请求的 `num_computed_tokens` 追平 `num_tokens_with_spec`。这统一了 chunked prefill、prefix caching、speculative decoding，以及未来的 "jump decoding"。

---

## 整体架构与文件一览

`vllm/v1/core/` 在物理上分为两个子目录层级：根目录下是 KV Cache 子系统，`sched/` 子目录下是调度子系统。

```
vllm/v1/core/
├── sched/                       # 调度子系统
│   ├── interface.py             # SchedulerInterface 抽象基类 + PauseState
│   ├── scheduler.py             # Scheduler 核心实现（~2200 行）
│   ├── async_scheduler.py       # AsyncScheduler 异步调度子类
│   ├── output.py                # SchedulerOutput / NewRequestData / CachedRequestData
│   ├── request_queue.py         # FCFS / Priority 两种请求队列
│   └── utils.py                 # check_stop、重复检测、remove_all
├── kv_cache_manager.py          # KVCacheManager 门面（Scheduler 唯一入口）
├── kv_cache_coordinator.py      # 多 KV group 协调器（单组/混合/禁用前缀缓存）
├── single_type_kv_cache_manager.py  # 各 attention 类型的块管理器（Full/SWA/Mamba...）
├── block_pool.py                # BlockPool：物理块池 + 前缀缓存哈希表
├── kv_cache_utils.py            # KVCacheBlock、空闲链表、块哈希、config 规划（~2100 行）
├── kv_cache_metrics.py          # 块生命周期采样与驱逐事件指标
└── encoder_cache_manager.py     # 多模态 encoder 输出缓存
```

调用分层（自顶向下）：

```
EngineCore.step()
   │  调用 schedule() / update_from_output()
   ▼
Scheduler (sched/scheduler.py)
   │  持有 KVCacheManager、EncoderCacheManager、RequestQueue
   ▼
KVCacheManager (kv_cache_manager.py)         ← Scheduler 只接触这一层
   │  持有 coordinator、block_pool
   ▼
KVCacheCoordinator (kv_cache_coordinator.py)
   │  持有唯一 BlockPool + 多个 SingleTypeKVCacheManager
   ├──► BlockPool (block_pool.py)            [物理块唯一所有者]
   └──► SingleTypeKVCacheManager[]           [每 attention group 一个]
            使用 KVCacheBlock / FreeKVCacheBlockQueue (kv_cache_utils.py)
```

关键设计原则：
- **物理块集中管理**：`BlockPool` 是所有 `KVCacheBlock` 对象的唯一所有者；所有 manager 共享同一个 `BlockPool` 实例。
- **块表按请求、按 group 分散**：每个 `SingleTypeKVCacheManager` 维护自己的 `req_to_blocks: request_id → list[KVCacheBlock]`。
- **门面隐藏内部结构**：`KVCacheManager` 对 Scheduler 只暴露 `KVCacheBlocks`（按 group 分组的元组），Scheduler 不直接接触 `BlockPool`。

---

## 一、核心数据结构：Request

在进入调度器之前，必须先理解它操作的对象——`Request`，定义在 [vllm/v1/request.py](../../vllm/v1/request.py)。

### 1.1 RequestStatus

[request.py:316-343](../../vllm/v1/request.py#L316-L343) 用 `IntEnum` 定义请求状态，并利用数值顺序编码"已完成"语义：

```python
class RequestStatus(enum.IntEnum):
    WAITING = auto()                                    # 1
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = auto()      # 2  等语法编译
    WAITING_FOR_REMOTE_KVS = auto()                     # 3  等远程 KV 加载
    WAITING_FOR_STREAMING_REQ = auto()                  # 4  等流式输入
    RUNNING = auto()                                    # 5
    PREEMPTED = auto()                                  # 6
    # 注意：PREEMPTED 之后的所有状态都被视为 finished
    FINISHED_STOPPED = auto()                           # 7
    FINISHED_LENGTH_CAPPED = auto()                     # 8
    FINISHED_ABORTED = auto()                           # 9
    FINISHED_IGNORED = auto()                           # 10
    FINISHED_ERROR = auto()                             # 11
    FINISHED_REPETITION = auto()                        # 12
```

- `is_finished(status)` 仅需一次比较 `status > RequestStatus.PREEMPTED`（[request.py:338](../../vllm/v1/request.py#L338)）。新增完成状态只要放在 `PREEMPTED` 之后即可，无需修改判断逻辑。
- 三个 `WAITING_FOR_*` 是"阻塞等待"状态，请求虽在等待队列但暂时不可调度，会被移入 `skipped_waiting` 队列。
- `_FINISHED_REASON_MAP`（[request.py:350-358](../../vllm/v1/request.py#L350-L358)）映射到 `FinishReason`；其中 `FINISHED_IGNORED`（prompt 超长）映射为 `LENGTH`，与 OpenAI API 行为一致。

### 1.2 Request 的关键字段

[request.py:59-188](../../vllm/v1/request.py#L59-L188) 中与调度/KV 强相关的字段：

| 字段 | 含义 |
|------|------|
| `request_id` / `client_index` / `priority` | 标识、输出回送目标、调度优先级（值越小越优先） |
| `arrival_time` | 到达时间，FCFS 与 priority 平局判定用 |
| `status` | `RequestStatus` |
| `sampling_params` / `pooling_params` | 生成/池化参数（二者互斥） |
| `max_tokens` | 生成模型取 `sampling_params.max_tokens`；pooling 固定为 1 |
| `num_prompt_tokens` | prompt 长度（token 或 embed） |
| `_output_token_ids` / `_all_token_ids` | 输出 token、prompt+输出全量；通过 `ConstantList` 暴露只读视图 |
| **`num_computed_tokens`** | **已计算（含前缀缓存命中）的 token 数——调度追赶模型的核心游标** |
| **`spec_token_ids`** | 投机解码的 draft tokens |
| `num_output_placeholders` | 异步调度中预留的输出占位符（1 真实 + N draft） |
| `block_hashes` | 该请求每个满块的内容哈希，用于前缀缓存查找 |
| `mm_features` | 多模态特征列表 |
| `num_preemptions` | 被抢占次数 |
| `resumable` / `streaming_queue` | 流式输入续传 |
| `lora_request` / `structured_output_request` | LoRA 与结构化输出 |

几个关键 property（[request.py:239-257](../../vllm/v1/request.py#L239-L257)）：

- `num_tokens = len(_all_token_ids)` —— 已确认的 prompt+output token 总数。
- `num_tokens_with_spec = num_tokens + len(spec_token_ids)` —— **含 draft token 的总长度，是 KV slot 分配的依据**。
- `num_output_tokens = len(_output_token_ids)`。

### 1.3 token 追加与块哈希联动

[request.py:217-233](../../vllm/v1/request.py#L217-L233)：

```python
def append_output_token_ids(self, token_ids):
    # 同时追加到 _output_token_ids 和 _all_token_ids，保持同步
    ...
    self.update_block_hashes()   # 增量计算新填满的块的哈希
```

`update_block_hashes()` 调用构造时注入的 `_block_hasher` 闭包，只为新产生的完整块计算哈希并 extend 到 `block_hashes`。注意 `_block_hasher` **不绑定 self**（[request.py:173-176](../../vllm/v1/request.py#L173-L176)），避免 `Request → partial → Request` 引用循环阻碍 CPython 引用计数即时回收。

### 1.4 优先级比较

[request.py:302-313](../../vllm/v1/request.py#L302-L313) 的 `__lt__` 定义优先队列排序：先比 `priority`（升序，值小优先），再比 `arrival_time`（早者优先），再比 `request_id`，最后用 `id(self)` 兜底，保证比较全序且稳定。

> **注意**：Request 对象本身**不持有 block_table**。物理块映射由 `KVCacheManager`/coordinator 以 `request_id → blocks` 维护。Request 只持有逻辑 `block_hashes` 用于前缀缓存查找。这种分离让 Request 保持轻量、可安全在异步调度中传递。

---

## 二、请求队列与调度策略 (`sched/request_queue.py`)

### 2.1 SchedulingPolicy

[request_queue.py:13-17](../../vllm/v1/core/sched/request_queue.py#L13-L17)：

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
```

由 `SchedulerConfig.policy`（[config/scheduler.py:109](../../vllm/config/scheduler.py#L109)）控制。

### 2.2 两种队列实现

| 实现 | 底层结构 | 入队 | 出队 | 前插 | remove |
|------|----------|------|------|------|--------|
| `FCFSRequestQueue` | `collections.deque` | `append` | `popleft` | `appendleft` | 转 list 过滤后重建，O(n) |
| `PriorityRequestQueue` | `heapq` 堆 | `heappush` | `heappop` | 等同 `add`（无"前插"语义） | `list.remove` + `heapify`，O(n) |

`PriorityRequestQueue.__iter__`（[request_queue.py:194-198](../../vllm/v1/core/sched/request_queue.py#L194-L198)）在遍历前**复制堆**再逐个 heappop 副本，保证按优先级顺序遍历且不破坏原堆。

`create_request_queue(policy)` 工厂函数（[request_queue.py:201-208](../../vllm/v1/core/sched/request_queue.py#L201-L208)）返回对应实例。调度器内部创建了**三个**队列：

- `waiting` —— 可调度的等待请求；
- `skipped_waiting` —— 因异步依赖（等远程 KV、等语法编译、等流式输入）被跳过的等待请求；
- `step_skipped_waiting` —— 每步临时收集本步新跳过的请求，调度结束后 prepend 到 `skipped_waiting` **前面**，保证新跳过的请求比旧跳过的优先重试（[scheduler.py:802-804](../../vllm/v1/core/sched/scheduler.py#L802-L804)）。

### 2.3 队列选择策略

调度 waiting 请求时调用 `_select_waiting_queue_for_scheduling()`（[scheduler.py:1529-1539](../../vllm/v1/core/sched/scheduler.py#L1529-L1539)）：
- **FCFS**：优先 `skipped_waiting`，因为被跳过的请求通常更早到达；
- **PRIORITY**：两队列都非空时比较队头，用 `Request.__lt__` 取优先级更高者。

---

## 三、调度器抽象接口与暂停状态 (`sched/interface.py`)

### 3.1 PauseState

[interface.py:22-33](../../vllm/v1/core/sched/interface.py#L22-L33)：

```python
class PauseState(IntEnum):
    UNPAUSED = 0    # 正常调度
    PAUSED_NEW = 1  # 只调度 running 中的请求，不接纳新请求
    PAUSED_ALL = 2  # 不调度任何请求（token_budget 置 0）
```

用于优雅关闭/模型重载。`PAUSED_NEW` 允许在途请求生成完毕但不接纳新请求；`PAUSED_ALL` 完全冻结调度。

### 3.2 SchedulerInterface

抽象基类（[interface.py:36-244](../../vllm/v1/core/sched/interface.py#L36-L244)）定义调度器契约。核心方法：

| 方法 | 职责 |
|------|------|
| `schedule()` | 返回一次前向的调度决策 `SchedulerOutput` |
| `update_from_output()` | 模型执行后用 `ModelRunnerOutput` 更新状态，返回各 client 的 `EngineCoreOutputs` |
| `get_grammar_bitmask()` | 获取结构化输出语法位掩码 |
| `update_draft_token_ids()` / `update_draft_token_ids_in_output()` | 注入投机解码 draft token |
| `add_request()` / `finish_requests()` | 入队 / 中止完成 |
| `reset_prefix_cache()` / `reset_encoder_cache()` | 缓存重置 |
| `get_num_unfinished_requests()` / `has_requests()` | 状态查询 |
| `make_stats()` / `shutdown()` | 统计与关闭 |

接口注释明确了调度的本质：**产生一个 `{req_id: num_tokens}` 字典，决定每个请求在本次前向中处理多少 token**。这统一了 prefill、decode、chunked prefill、prefix caching、speculative decoding。

调度器实现类由 `SchedulerConfig.get_scheduler_cls()`（[config/scheduler.py:168-188](../../vllm/config/scheduler.py#L168-L188)）工厂选择：`async_scheduling=True` 时返回 `AsyncScheduler`，否则返回 `Scheduler`。

---

## 四、Scheduler 核心 (`sched/scheduler.py`)

### 4.1 初始化要点

`Scheduler.__init__`（[scheduler.py:63-258](../../vllm/v1/core/sched/scheduler.py#L63-L258)）的关键状态：

**调度约束**（[scheduler.py:100-111](../../vllm/v1/core/sched/scheduler.py#L100-L111)）：
```python
self.max_num_running_reqs = scheduler_config.max_num_seqs
self.max_num_scheduled_tokens = (
    scheduler_config.max_num_scheduled_tokens
    if scheduler_config.max_num_scheduled_tokens
    else scheduler_config.max_num_batched_tokens
)
self.max_model_len = model_config.max_model_len
```

**核心队列状态**（[scheduler.py:152-175](../../vllm/v1/core/sched/scheduler.py#L152-L175)）：
```python
self.requests: dict[str, Request] = {}          # req_id -> Request（所有活跃请求）
self.policy = SchedulingPolicy(...)
self.waiting = create_request_queue(self.policy)
self.skipped_waiting = create_request_queue(...)
self.running: list[Request] = []                 # 运行队列（list，非队列）
self.finished_req_ids: set[str] = set()
```

**子组件**：
- `KVCacheManager`（[scheduler.py:220-241](../../vllm/v1/core/sched/scheduler.py#L220-L241)）——传入 `max_model_len`、`max_num_batched_tokens`、prefix caching 开关、EAGLE 开关、DCP/PCP world size、hash_block_size 等；
- `EncoderCacheManager` 或 `EncoderDecoderCacheManager`（[scheduler.py:200-206](../../vllm/v1/core/sched/scheduler.py#L200-L206)）；
- `KVConnectorBase_V1`（role=SCHEDULER）——用于 P/D 分离和 KV offload；
- 投机解码：`use_eagle`、`num_spec_tokens`、`num_lookahead_tokens`（EAGLE/draft 时 lookahead = spec_tokens，用于 KV block 预分配）。

### 4.2 `schedule()` 核心算法

这是整个模块的心脏。算法分为两个阶段：**先调度 RUNNING 请求，再（仅当本步无抢占时）调度 WAITING 请求**。

#### 阶段 0：初始化（[scheduler.py:322-343](../../vllm/v1/core/sched/scheduler.py#L322-L343)）

```python
token_budget = self.max_num_scheduled_tokens
if self._pause_state == PauseState.PAUSED_ALL:
    token_budget = 0
encoder_compute_budget = self.max_num_encoder_input_tokens
self.kv_cache_manager.new_step_starts()   # 通知 KV 管理器新步骤开始（Mamba 清空本步缓存集合）
```

#### 阶段 1：调度 RUNNING 请求（[scheduler.py:345-513](../../vllm/v1/core/sched/scheduler.py#L345-L513)）

对 `self.running` 按索引遍历，直到预算耗尽：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

这就是"追赶"公式：待计算量 = 目标长度（含 draft 和占位符）− 已计算量。随后依次施加三道截断：
1. `long_prefill_token_threshold`（长 prompt 分块阈值，0 表示不限）；
2. `token_budget`（本步剩余预算）；
3. `max_model_len - 1 - num_computed_tokens`（不超模型长度，spec decoding 时必要）。

然后：
- 若请求有编码器输入，调 `_try_schedule_encoder_inputs()`，可能缩小 `num_new_tokens` 并消耗编码器预算；
- Mamba 模型按需做 block 对齐（`_mamba_block_aligned_split`）；
- **`num_new_tokens == 0` 时用 `continue`（不是 break）**——注释明确这**不严格遵循 FCFS**，允许后续请求被调度。原因包括 PP 中 prompt 已调度但未完成、异步调度达上限、编码器预算耗尽、Mamba 对齐块不足。

**KV block 分配与抢占循环**（[scheduler.py:422-472](../../vllm/v1/core/sched/scheduler.py#L422-L472)）：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens,
        num_lookahead_tokens=self.num_lookahead_tokens,
    )
    if new_blocks is not None:
        break
    # 分配失败 → 抢占
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running,
                           key=lambda r: (r.priority, r.arrival_time))
        # 从 running 移除；若已在本步调度，回滚其全部记账...
    else:
        preempted_req = self.running.pop()   # FCFS: LIFO，弹出最后加入的
    self._preempt_request(preempted_req, scheduled_timestamp)
    preempted_reqs.append(preempted_req)
    if preempted_req == request:
        break   # 抢占到自己仍无法分配，退出
```

抢占策略差异：
- **FCFS**：`self.running.pop()` 弹出**最后加入**的请求（LIFO 抢占），保护更早到达的请求；
- **PRIORITY**：用 `max(key=(priority, arrival_time))` 选**优先级最低**（priority 值最大）且到达最晚的。若被抢占者已在本步 earlier 被调度，需完整回滚：从 `scheduled_running_reqs` 移除、返还 token_budget、移除 blocks、spec tokens、编码器预算，并 `req_index -= 1`。

抢占后 `allocate_slots` 重试。若抢占到自己仍返回 None（资源极度紧张），break 退出 running 调度循环。

分配成功后记录 `num_scheduled_tokens[req_id]`、扣减预算、处理 spec token，并为调度的编码器输入分配 encoder cache。

#### 阶段 2：调度 WAITING 请求（[scheduler.py:525-804](../../vllm/v1/core/sched/scheduler.py#L525-L804)）

**前置条件**：`not preempted_reqs and self._pause_state == PAUSED_STATE.UNPAUSED`（[scheduler.py:526](../../vllm/v1/core/sched/scheduler.py#L526)）。**一旦本步发生抢占，就不接纳新请求**——优先让被抢占请求重新 prefill，防止饥饿。

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break
```

每个等待请求的处理：

1. **选队列并 peek 队头**：按 `_select_waiting_queue_for_scheduling()` 选 `waiting` 或 `skipped_waiting`。若处于三种 blocked 状态，尝试 `_try_promote_blocked_waiting_request`；仍不可调度则移入 `step_skipped_waiting` 并 continue。
2. **LoRA 约束**：若已达 `max_loras` 且新请求的 LoRA 不在已调度集合中，跳过。
3. **查询前缀缓存命中**（首次调度，`num_computed_tokens == 0`）：
   - 本地命中：`kv_cache_manager.get_computed_blocks(request)` 返回命中的 blocks 和 token 数；
   - 外部命中：若有 KV connector，`connector.get_num_new_matched_tokens()` 返回远程匹配 token 数（返回 None 表示暂不确定，跳过该请求）；
   - `num_computed_tokens = local + external`。
4. **计算 `num_new_tokens = request.num_tokens - num_computed_tokens`**。注意用 `num_tokens` 而非 `num_prompt_tokens`，因为恢复的请求可能已有 output token。
5. 截断：`long_prefill_token_threshold` → **chunked prefill 检查** → `token_budget`。
   - 关键：若**未启用 chunked prefill** 且 `num_new_tokens > token_budget`，用 **break**（不是 continue）——FCFS 下后续请求也无法在剩余预算内容纳，应停止。
6. **分配 KV slots**：
   ```python
   new_blocks = self.kv_cache_manager.allocate_slots(
       request, num_new_tokens,
       num_new_computed_tokens=num_new_local_computed_tokens,
       new_computed_blocks=new_computed_blocks,
       num_lookahead_tokens=effective_lookahead_tokens,  # 首次调度为 0
       num_external_computed_tokens=num_external_computed_tokens,
       delay_cache_blocks=load_kv_async,
       num_encoder_tokens=num_encoder_tokens,
       full_sequence_must_fit=self.scheduler_reserve_full_isl,
   )
   ```
   waiting 请求分配失败**不触发抢占**（running 阶段已耗尽资源，抢占刚加入的请求无意义），直接 break。
7. 异步 KV 加载路径：若 `load_kv_async`，`num_new_tokens=0`，请求状态置 `WAITING_FOR_REMOTE_KVS` 放入 skipped 队列，**不加入 running**。
8. 否则 `self.running.append(request)`，并按原状态分类：`WAITING` → `scheduled_new_reqs`；`PREEMPTED` → `scheduled_resumed_reqs`。
9. 设置 `request.status = RUNNING`、`request.num_computed_tokens`、扣减预算、分配编码器缓存。

> **`scheduler_reserve_full_isl` 准入闸门**（[scheduler.py:140-144](../../vllm/config/scheduler.py#L140-L144)）：默认 True。分配前检查**完整输入序列长度**（而非仅第一个 chunk）能否装入 KV cache，防止 chunked prefill 下过度接纳导致 KV cache 抖动。

#### 阶段 3：后置处理与输出构造（[scheduler.py:806-903](../../vllm/v1/core/sched/scheduler.py#L806-L903)）

- **约束断言**：`total_num_scheduled_tokens ≤ max_num_scheduled_tokens`、`token_budget ≥ 0`、`len(running) ≤ max_num_running_reqs`；
- **公共前缀计算**：`kv_cache_manager.get_num_common_prefix_blocks()` 取所有 running 请求的最长公共前缀 block 数，用于 cascade attention；
- 构造 `NewRequestData`（首次调度）和 `CachedRequestData`（running/resumed 增量）；
- 若 `needs_kv_cache_zeroing`，收集本步新分配 block  IDs 到 `new_block_ids_to_zero`（Mamba 等需清零防 NaN）；
- 构造 `SchedulerOutput` 和 KV/EC connector 元数据；
- 最后调 **`_update_after_schedule()`** 推进 `num_computed_tokens`。

#### `continue` vs `break` 的设计

这是阅读调度器时最需要注意的细节：

| 场景 | 控制流 | 原因 |
|------|--------|------|
| Running 请求 `num_new_tokens==0` | `continue` | 允许后续请求调度，非严格 FCFS |
| Waiting 请求未启用 chunked prefill 且超预算 | `break` | FCFS 下后续也装不下 |
| Waiting 请求 KV block 分配失败 | `break` | 资源耗尽，继续无意义 |
| Running 请求抢占到自己 | `break` | 无法调度，退出 running 循环 |

### 4.3 抢占：`_preempt_request`

[scheduler.py:910-930](../../vllm/v1/core/sched/scheduler.py#L910-L930)：

```python
def _preempt_request(self, request, timestamp):
    assert request.status == RequestStatus.RUNNING
    self.kv_cache_manager.free(request)          # 释放 KV
    self.encoder_cache_manager.free(request)     # 释放编码器缓存
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0              # 重置，重算全部
    if request.spec_token_ids:
        request.spec_token_ids = []
    request.num_preemptions += 1
    request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
    self.waiting.prepend_request(request)        # 放入 waiting 头部，优先恢复
```

抢占是**重算式（recompute）**而非交换式：`num_computed_tokens` 归零，KV 全部释放，请求回到 waiting 队首优先恢复。调用者负责从 `running` 列表移除。

### 4.4 延迟推进 computed tokens

`_update_after_schedule()`（[scheduler.py:932-956](../../vllm/v1/core/sched/scheduler.py#L932-L956)）在 **schedule() 末尾**（而非 `update_from_output` 中）推进每个请求的 `num_computed_tokens`：

```python
for req_id, num_scheduled_token in num_scheduled_tokens.items():
    request = self.requests[req_id]
    request.num_computed_tokens += num_scheduled_token
    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders
    )
```

注释（[scheduler.py:933-941](../../vllm/v1/core/sched/scheduler.py#L933-L941)）解释这样设计的三个好处：
1. `SchedulerOutput` 包含原始调度 token 数以确定 input IDs；
2. 下一步可立即再次调度 prefill 请求（连续 chunked prefill）；
3. 若 spec token 被拒绝，在 `update_from_output` 中回退。

> 注意 `self.finished_req_ids` 在末尾用**赋值替换为空集合**而非 `clear()`（[scheduler.py:956](../../vllm/v1/core/sched/scheduler.py#L956)），因为旧集合已被构造好的 `SchedulerOutput` 引用，clear 会污染输出。

### 4.5 `update_from_output()`：模型执行后更新

[scheduler.py:1248-1513](../../vllm/v1/core/sched/scheduler.py#L1248-L1513) 是模型前向后的状态更新主流程，返回按 `client_index` 分组的 `EngineCoreOutputs`。

主循环（遍历 `num_scheduled_tokens`）：
1. **投机解码接受/拒绝**：`num_accepted = len(generated_token_ids) - 1`，`num_rejected = num_draft_tokens - num_accepted`；回退 `num_computed_tokens` 和 `num_output_placeholders`；
2. 释放已处理的编码器输入；
3. **停止检查**：
   - 有新 token → `_update_request_with_output()`（逐个追加 token 并调 `check_stop`）；
   - 无 token 但有 `pooler_output` → pooling 请求直接标记完成；
4. 结构化输出语法校验：grammar 拒绝 token 则标记 `FINISHED_ERROR`；
5. **停止处理**：捕获 `finish_reason`，调 `_handle_stopped_request`；真正完成则 `_free_request`；
6. 提取 logprobs、NaN 统计，构造 `EngineCoreOutput`；
7. partial prefill（未生成新 token）不产生输出。

随后用 `remove_all` 从 running/waiting 清理停止的请求，处理 KV 加载失败策略（`recompute` 重算 vs `fail` 报错），收集并发布 KV cache 事件，最后构造按 client 分组的 `EngineCoreOutputs`（stats 附加到第一个 client，即使无请求输出也创建空输出放 stats）。

### 4.6 停止判定 `check_stop`

[sched/utils.py:94-130](../../vllm/v1/core/sched/utils.py#L94-L130) 是请求停止判定的核心，按优先级：

1. **min_tokens 未达**：`num_output_tokens < min_tokens` 时直接返回 False，不触发任何停止；
2. **EOS token**：最后一个 token == `eos_token_id` → `FINISHED_STOPPED`；
3. **stop_token_ids**：最后一个 token 在停止集合中 → `FINISHED_STOPPED`，记录 `stop_reason`；
4. **长度上限**：`num_tokens >= max_model_len` 或 `num_output_tokens >= max_tokens` → `FINISHED_LENGTH_CAPPED`；
5. **重复检测**：`check_sequence_repetition`（[utils.py:28-59](../../vllm/v1/core/sched/utils.py#L28-L59)）检测尾部重复模式 → `FINISHED_REPETITION`（幻觉/无循环退化检测）；
6. 都不满足返回 False。

该函数断言 `not request.pooling_params`——pooling 请求在调用方单独处理（有 pooler_output 即停）。

### 4.7 请求生命周期管理

- **`add_request()`**（[scheduler.py:1665-1685](../../vllm/v1/core/sched/scheduler.py#L1665-L1685)）：若 request_id 已存在，作为流式 update 处理（追加到 `streaming_queue`）；否则创建 `Request`、入队、记入 `self.requests`。
- **`finish_requests()`**（[scheduler.py:1687-1748](../../vllm/v1/core/sched/scheduler.py#L1687-L1748)）：支持单个 ID、ID 列表或 None（全部）；两阶段——先收集并从队列移除，再设置状态并释放。`WAITING_FOR_REMOTE_KVS` 的请求可能延迟释放 block（等异步接收完成）。
- **`_free_request()` / `_free_blocks()`**（[scheduler.py:1750-1771](../../vllm/v1/core/sched/scheduler.py#L1750-L1771)）：调 connector 的 `request_finished`、释放编码器缓存、记录 `finished_req_ids`、释放 KV block 并从 `self.requests` 删除。
- **`reset_prefix_cache()`**（[scheduler.py:1795-1850](../../vllm/v1/core/sched/scheduler.py#L1795-L1850)）：若需重置 running 请求，**逆序**抢占所有 running（逆序保证恢复时 FIFO 顺序），再调 `kv_cache_manager.reset_prefix_cache()`。

---

## 五、AsyncScheduler (`sched/async_scheduler.py`)

异步调度器只有 61 行，通过覆写两个方法实现"调度与执行解耦"，用于消除 GPU 气泡。

### 5.1 placeholder 机制

`_update_after_schedule()`（[async_scheduler.py:18-35](../../vllm/v1/core/sched/async_scheduler.py#L18-L35)）在推进 computed tokens 后，为每个 decode 请求增加输出占位符：

```python
request.num_output_placeholders += 1 + cur_num_spec_tokens
request.spec_token_ids = self._spec_token_placeholders  # [-1] * num_spec_tokens
```

这表示本步将生成 **1 个真实 token + N 个 draft token**。占位符让调度器在**还不知道实际 token ID** 时就预留 KV slot，从而调度和执行可以流水线重叠。

### 5.2 收到输出后缓存 blocks

`_update_request_with_output()`（[async_scheduler.py:37-60](../../vllm/v1/core/sched/async_scheduler.py#L37-L60)）覆写父类：
1. `discard_latest_async_tokens` 标志（reset_prefix_cache 强制抢占场景）丢弃最新异步 token；
2. 调父类追加 token 和检查停止；
3. 扣减 `num_output_placeholders -= len(new_token_ids)`；
4. **在收到输出后立即缓存新 token 的 KV blocks**（位置 `num_computed_tokens - num_output_placeholders`）。

这是与同步调度的根本区别：**同步在 schedule 时分配 blocks，异步在收到输出后缓存 blocks**。

---

## 六、调度输出结构 (`sched/output.py`)

Scheduler 与 Worker 之间的通信契约，全部用 dataclass 定义。

### 6.1 NewRequestData（[output.py:30-108](../../vllm/v1/core/sched/output.py#L30-L108)）

首次调度的请求数据，发送给 worker 并缓存。字段包括 `req_id`、`prompt_token_ids`、`mm_features`、采样/池化参数、`block_ids`（每 KV group 一个 list）、`num_computed_tokens`（含前缀缓存命中）、LoRA、prompt embeds 等。`from_request()` 工厂方法构造；v2 model runner 额外传 `prefill_token_ids`。

### 6.2 CachedRequestData（[output.py:111-177](../../vllm/v1/core/sched/output.py#L111-L177)）

已调度过请求的**增量**数据：

| 字段 | 说明 |
|------|------|
| `req_ids` | running + resumed 顺序的请求 ID |
| `resumed_req_ids` | 被抢占恢复的请求——其 `new_block_ids` 是**全量替换** block table，其他请求是**追加** |
| `new_token_ids` | **仅流水线并行（PP）使用**，跨 stage 回传采样 token |
| `new_block_ids` | 本步新分配的 block |
| `num_computed_tokens` / `num_output_tokens` | 每请求调度前的已计算/输出 token 数（含 placeholders） |

`is_context_phase(req_id)`（[output.py:163-165](../../vllm/v1/core/sched/output.py#L163-L165)）：`num_output_tokens == 0` 表示仍在 prefill/context 阶段。

> **增量通信优化**：首次调度发完整 `NewRequestData` 并在 worker 缓存，后续只发 `CachedRequestData` 增量，显著降低调度器→worker 的通信量。

### 6.3 SchedulerOutput（[output.py:180-255](../../vllm/v1/core/sched/output.py#L180-L255)）

每步一个实例，关键字段：

| 字段 | 说明 |
|------|------|
| `scheduled_new_reqs` / `scheduled_cached_reqs` | 新请求 / 增量请求数据 |
| **`num_scheduled_tokens`** | **核心字段**：每请求本步调度 token 数 |
| `total_num_scheduled_tokens` | 总和，必须 ≤ `max_num_scheduled_tokens` |
| `scheduled_spec_decode_tokens` | 投机解码 draft tokens |
| `scheduled_encoder_inputs` | 需编码器处理的多模态输入索引 |
| `num_common_prefix_blocks` | 所有 running 请求的最长公共前缀 block 数（每 group），cascade attention 用 |
| `finished_req_ids` | 上步到本步完成的请求，通知 worker 释放缓存 |
| `free_encoder_mm_hashes` | 需从编码器缓存释放的多模态 hash |
| `preempted_req_ids` | 本步被抢占的请求（**仅 v2 model runner**） |
| `new_block_ids_to_zero` | 新分配需清零的 block（防 NaN/脏数据） |
| `kv_connector_metadata` / `ec_connector_metadata` | KV/编码器连接器元数据 |
| `num_invalid_spec_tokens` | 被语法拒绝的 draft token 数（用 -1 填充） |

### 6.4 GrammarOutput（[output.py:258-263](../../vllm/v1/core/sched/output.py#L258-L263)）

结构化输出的语法位掩码：`structured_output_request_ids` + `grammar_bitmask`（numpy int32 数组，行顺序与请求 ID 对应）。

---

## 七、KV Cache 子系统

理解了调度器"要多少 token"，接下来看 KV Cache 子系统"把它们放在哪、复用谁"。

### 7.1 KVCacheSpec 类型体系

块管理器的行为由 `KVCacheSpec`（[vllm/v1/kv_cache_interface.py](../../vllm/v1/kv_cache_interface.py)）决定，它描述一类层的 KV cache 格式（block_size、页大小、内存上限）。核心继承体系：

```
KVCacheSpec
├── AttentionSpec (num_kv_heads, head_size, dtype, kv_quant_mode)
│   ├── FullAttentionSpec            全注意力（含 sliding_window 字段时表示混合模型中按全注意力分配）
│   │   ├── TQFullAttentionSpec      TQ 感知页大小
│   │   ├── MLAAttentionSpec         DeepSeek MLA（compress_ratio、自定义 fp8 布局）
│   │   └── SinkFullAttentionSpec   Attention sink（永久驻留块）
│   ├── SlidingWindowSpec           滑动窗口注意力
│   │   └── SlidingWindowMLASpec    滑动窗口 + MLA 格式
│   ├── ChunkedLocalAttentionSpec  分块局部注意力
│   ├── EncoderOnlyAttentionSpec   编码器层（不需 KV cache）
│   └── CrossAttentionSpec         encoder-decoder 交叉注意力
├── MambaSpec                       Mamba/SSM 状态缓存
└── UniformTypeKVCacheSpecs         多层同类型但 hidden size 不同（DeepSeekV4）
```

`KVCacheGroupSpec`（[kv_cache_interface.py:757-768](../../vllm/v1/kv_cache_interface.py#L757-L768)）把**共享同一 block table 的层**归为一个 group，作为块管理器的一个管理单元。一个模型可以有多个 group（如 full attention + sliding window 混合模型）。

`KVCacheConfig`（[kv_cache_interface.py:772-796](../../vllm/v1/kv_cache_interface.py#L772-L796)）是模型的 KV cache 配置：`num_blocks`（总块数）、`kv_cache_tensors`（worker 如何初始化张量）、`kv_cache_groups`（group 列表）。

> 物理块大小在所有 group 间必须一致（由 `unify_kv_cache_spec_page_size` 保证），这是它们能共享同一个 `BlockPool` 的硬约束。

### 7.2 物理块元数据：KVCacheBlock（`kv_cache_utils.py`）

[kv_cache_utils.py:113-159](../../vllm/v1/core/kv_cache_utils.py#L113-L159) 的 `KVCacheBlock` 用 `@dataclass(slots=True)` 定义，是块的**纯元数据**（不含张量，张量由 model runner 管理）：

| 字段 | 含义 |
|------|------|
| `block_id` | 0 ~ num_gpu_blocks-1，即 `blocks` 列表下标 |
| `ref_cnt` | 引用计数 |
| `_block_hash` | 仅当块满且被前缀缓存时非 None |
| `prev_free_block` / `next_free_block` | 空闲链表指针 |
| `is_null` | 是否为空占位块（block 0，永不缓存/释放） |

`block_hash` 的 setter **只允许从 None 设为非 None**（[kv_cache_utils.py:137-142](../../vllm/v1/core/kv_cache_utils.py#L137-L142)）：块一旦缓存并打哈希，不能直接覆盖，必须先 `reset_hash()`，防止状态错误。

### 7.3 零分配空闲链表：FreeKVCacheBlockQueue

[kv_cache_utils.py:162-370](../../vllm/v1/core/kv_cache_utils.py#L162-L370) 是自定义双向链表，**直接操作块对象的 `prev/next_free_block` 指针，不分配任何 Python 对象**，性能接近 C deque，同时支持 O(1) 中间删除。

- fake head/tail（block_id=-1）保证每个真实块总有 prev/next，减少分支；
- `popleft()` / `popleft_n(n)` 从头部（LRU 最旧端）分配；
- `remove(block)` O(1) 从中间摘除（前缀缓存命中时从空闲队列摘除）；
- `append(block)` / `append_n(blocks)` 追加到尾部（LRU 最新端）。

LRU 顺序由释放时的顺序维护：调用方 `free()` 时传入 `reversed(req_blocks)`，使块链**尾部（hash 更多、复用价值更低）先进入空闲队列头部、先被驱逐**。

### 7.4 BlockPool：物理块唯一所有者（`block_pool.py`）

[block_pool.py:130-509](../../vllm/v1/core/block_pool.py#L130-L509) 持有所有 `KVCacheBlock`、空闲队列和前缀缓存哈希表。

**构造**（[block_pool.py:149-182](../../vllm/v1/core/block_pool.py#L149-L182)）：
- `blocks: list[KVCacheBlock]`，block_id 即下标；
- `free_block_queue = FreeKVCacheBlockQueue(self.blocks)`，初始所有块空闲；
- `cached_block_hash_to_block: BlockHashToBlockMap`；
- **`null_block = self.free_block_queue.popleft()`：block 0 被永久征用为 null_block**（`is_null=True`），因此实际可用块为 `num_gpu_blocks - 1`。`get_usage()` 的分母也排除它。

**核心方法**：

| 方法 | 行为 |
|------|------|
| `get_cached_block(block_hash, group_ids)` | 逐 group 拼 `(block_hash, group_id)` 查哈希表；任一 miss 返回 None |
| `cache_full_blocks(...)` | 满块注册到前缀缓存：设 `block_hash`、插入哈希表、生成 `BlockStored` 事件 |
| `get_new_blocks(n)` | 从空闲队列 `popleft_n`；若块带 hash（驱逐候选）先 `_maybe_evict_cached_block`；`ref_cnt=1` |
| `touch(blocks)` | 前缀缓存命中：ref_cnt=0 的块从空闲队列摘除，`ref_cnt++` |
| `free_blocks(ordered_blocks)` | 两遍：先所有块 `ref_cnt--`；再把 ref_cnt 归 0 的非 null 块 append 回空闲队列尾部 |
| `evict_blocks(block_ids)` | 按 ID 强制驱逐（供 KV connector）；ref_cnt>0 的块只从哈希表移除，不释放 |
| `reset_prefix_cache()` | 仅当只剩 null_block 在用时才能重置（否则警告返回 False） |

**关键：引用计数而非 COW**。vLLM v1 **不使用传统 copy-on-write**：
- 多个请求的块表可引用同一个 `KVCacheBlock` 对象（前缀缓存命中后 `touch()` 增 ref_cnt）；
- 块内容不可变（满并缓存后 hash 不可改），追加新 token 总是分配**新块**，不修改共享块；
- 因此天然无需 COW：共享只读块 + 新块追加。ref_cnt 归 0 时块才真正可驱逐。

**BlockHashToBlockMap**（[block_pool.py:34-127](../../vllm/v1/core/block_pool.py#L34-L127)）用联合类型优化 GC：单块时值就是 `KVCacheBlock`；哈希冲突（不同 block_id 同内容哈希）时升级为 `dict[block_id, KVCacheBlock]`。当前**不做去重**（注释 L48-52），以保证分配的 block ID 不变，使 block table 只追加（append-only）。

**块状态转换**：

```
   创建 ──► Free, no hash (ref_cnt=0, 在 free_queue, hash=None)
                   │ get_new_blocks (popleft, ref_cnt=1)
                   ▼
            Allocated (ref_cnt>0, 不在 free_queue)
                   │ cache_full_blocks (设 block_hash)
                   ▼
       Allocated + Cached (ref_cnt>0, hash set)
                   │ free_blocks (ref_cnt→0)
                   ▼
       Cached, evictable (ref_cnt=0, 在 free_queue, hash set)
                   │ touch (另一请求命中): ref_cnt 0→1, 摘除队列 → 回到 Allocated+Cached
                   │ get_new_blocks (popleft → 驱逐: pop 哈希表, reset_hash)
                   ▼
             Free, no hash (循环)

   null_block (block 0): is_null=True，永不参与上述转换
```

### 7.5 前缀缓存哈希机制（`kv_cache_utils.py`）

**哈希链**是前缀缓存的基础。每个满块的哈希 = `hash(parent_block_hash, token_ids, extra_keys)`（[kv_cache_utils.py:539-566](../../vllm/v1/core/kv_cache_utils.py#L539-L566)）。第一块的 parent 用全局 `NONE_HASH`。相同 token 序列必然产生相同哈希链，因此能按块粒度匹配前缀。

`BlockHash` 是 `NewType("BlockHash", bytes)`；`BlockHashWithGroupId` 在末尾拼 4 字节大端 group id（[kv_cache_utils.py:37-72](../../vllm/v1/core/kv_cache_utils.py#L37-L72)），使不同 KV group 的同名哈希块能存在同一字典里而不冲突，同时避免 tuple 分配。

**extra_keys** 确保不同上下文的相同 token 不会错误命中（`generate_block_hash_extra_keys`，[kv_cache_utils.py:501-536](../../vllm/v1/core/kv_cache_utils.py#L501-L536)）：
- 多模态：`(mm_identifier, offset_in_block)`；
- LoRA：adapter name；
- `cache_salt`：仅首块；
- prompt embeds：按 `(start,end)` 范围做 sha256，缓存在 `request._prompt_embeds_per_block_hashes` 避免重复计算。

`get_request_block_hasher(block_size, caching_hash_fn)`（[kv_cache_utils.py:635-686](../../vllm/v1/core/kv_cache_utils.py#L635-L686)）返回闭包，**增量**为请求计算新填满块的哈希（`start_token_idx = len(block_hashes) * block_size`），不足一个完整块时早停。这就是 `Request.update_block_hashes()` 实际调用的函数。

**哈希粒度对齐**：单 group 时 `hash_block_size == block_size`；混合 group 时 hash_block_size 可以更小（GCD），通过 `BlockHashListWithBlockSize`（[kv_cache_utils.py:2056-2123](../../vllm/v1/core/kv_cache_utils.py#L2056-L2123)）惰性拼接连续细粒度哈希为粗粒度块哈希。`resolve_kv_cache_block_sizes`（[kv_cache_utils.py:569-632](../../vllm/v1/core/kv_cache_utils.py#L569-L632)）负责计算这两个尺寸。

### 7.6 单类型块管理器（`single_type_kv_cache_manager.py`）

每个 attention group 一个 `SingleTypeKVCacheManager`，维护该 group 的 `req_to_blocks: defaultdict[str, list[KVCacheBlock]]`。抽象基类（[single_type_kv_cache_manager.py:30-443](../../vllm/v1/core/single_type_kv_cache_manager.py#L30-L443)）定义通用流程，子类通过覆盖两个钩子差异化：

- `get_num_skipped_tokens(num_computed_tokens)` —— 多少旧 token 已不在注意力窗口内，其块可回收（FullAttention 返回 0）；
- `find_longest_cache_hit(...)`（classmethod）—— 如何在块哈希序列中找最长前缀命中。

**核心方法 `get_num_blocks_to_allocate`**（[single_type_kv_cache_manager.py:88-167](../../vllm/v1/core/single_type_kv_cache_manager.py#L88-L167)）：
- Running 请求快速路径：`max(num_required_blocks - num_req_blocks, 0)`；
- 新请求路径：`num_new_blocks = max(num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks), 0)`，再加上 evictable computed blocks（这些块在 free queue 中，touch 时会摘除，计入可用容量）。
- SlidingWindow/ChunkedLocal 有 `_max_admission_blocks_per_request` 上限（由 spec 的 `max_admission_blocks_per_request` 计算），仅在全序列准入检查（`apply_admission_cap=True`）时启用，避免 issue #39734 死锁。

**`allocate_new_computed_blocks`**（[L169-240](../../vllm/v1/core/single_type_kv_cache_manager.py#L169-L240)）：把前缀缓存命中的块加入请求——`block_pool.touch()` 增引用，skipped 位置填 `_null_block` 占位，再追加剩余 computed blocks。

**`free(request_id)`**（[L303-318](../../vllm/v1/core/single_type_kv_cache_manager.py#L303-L318)）：`reversed(req_blocks)` 逆序释放，让尾部块先回 LRU 尾部。

**`remove_skipped_blocks`**（[L385-426](../../vllm/v1/core/single_type_kv_cache_manager.py#L385-L426)）：滑动窗口/ChunkedLocal/Mamba 回收窗口外块——从后向前把 skipped 块替换为 null_block 并释放。

#### 各子类差异

| 管理器 | 对应 Spec | 前缀命中查找 | skipped tokens |
|--------|-----------|--------------|----------------|
| `FullAttentionManager` | FullAttention/MLA/TQ | **从左到右**遍历，miss 即 break | 0（不回收） |
| `SlidingWindowManager` | SlidingWindow | **从右向左**找窗口内连续 `cdiv(window-1, block_size)` 个块，左侧填 null | `max(0, computed - window + 1)` |
| `ChunkedLocalAttentionManager` | ChunkedLocal | 当前 chunk 左边界之前全填 null，从边界向右查 | `(computed // chunk_size) * chunk_size` |
| `MambaManager` | Mamba | **从右向左**只取最后一个命中块（SSM 只需最后状态），前插 null 对齐长度 | `computed - 1` |
| `CrossAttentionManager` | CrossAttention | 抛 NotImplementedError（encoder 状态每请求唯一，无复用） | — |
| `SinkFullAttentionManager` | SinkFullAttention | 同 FullAttention | 0；启动时永久征用 sink 块驻留 |

工厂映射 `spec_manager_map`（[single_type_kv_cache_manager.py:1142-1153](../../vllm/v1/core/single_type_kv_cache_manager.py#L1142-L1153)）决定每种 spec 用哪个 manager。

**几个值得注意的子类行为**：
- **FullAttentionManager 的 EAGLE 处理**（[L484-487](../../vllm/v1/core/single_type_kv_cache_manager.py#L484-L487)）：命中后弹出最后一个匹配块强制重算，以获取 eagle drafting head 所需的 hidden states。
- **MambaManager 同步块依赖防护**（[L903-911](../../vllm/v1/core/single_type_kv_cache_manager.py#L903-L911)）：若 new_computed_blocks 最后一块的 hash 在 `cached_blocks_this_step` 中（同一步其他请求刚生成），返回 `num_gpu_blocks + 1` 强制调度器认为块不足、推迟到下一步——Mamba 不能依赖同一步其他请求刚生成的状态。
- **Mamba align 模式**（`mamba_cache_mode == "align"`）：老请求最多再分配 1 块并复用之前的 speculative blocks，新请求分配 `1 + num_speculative_blocks` 块，以启用 Mamba 状态缓存。
- **get_num_common_prefix_blocks**：只有 FullAttention 返回非零值（遍历块，若 `block.ref_cnt == len(req_to_blocks)` 即所有持有者共享则计数，否则 break，downward-closed）；SWA/ChunkedLocal/Mamba/CrossAttention 都返回 0（cascade attention 不支持）。

### 7.7 协调器（`kv_cache_coordinator.py`）

`KVCacheCoordinator` 抽象基类（[kv_cache_coordinator.py:28-273](../../vllm/v1/core/kv_cache_coordinator.py#L28-L273)）持有：
- **唯一的 BlockPool**（L50-56）；
- `single_type_managers`：每个 kv_cache_group 一个 manager，全部共享同一 block_pool；
- `eagle_group_ids`：标记含 EAGLE draft 层的 group。

它把块分配/释放/缓存/查找等操作**逐 manager 分发**并汇总结果。工厂函数 `get_kv_cache_coordinator`（[kv_cache_coordinator.py:594-642](../../vllm/v1/core/kv_cache_coordinator.py#L594-L642)）按情况三选一：

| 协调器 | 适用场景 | 特点 |
|--------|----------|------|
| `KVCacheCoordinatorNoPrefixCache` | 禁用前缀缓存 | `find_longest_cache_hit` 永远返回空；`get_num_common_prefix_blocks` 返回全 0 |
| `UnitaryKVCacheCoordinator` | 单 group | 直接委托 manager[0]；断言 `hash_block_size == block_size` |
| `HybridKVCacheCoordinator` | 多 group 混合（如 full+SWA） | 用**迭代不动点算法**对齐不同 block_size group 的命中长度 |

**HybridKVCacheCoordinator 的不动点算法**（[kv_cache_coordinator.py:487-591](../../vllm/v1/core/kv_cache_coordinator.py#L487-L591)）：
- 每个 attention type 要么接受当前候选命中长度，要么缩短它；任一缩短则重启检查；长度单调下降有下界 0，必收敛。
- `FullAttentionSpec` 排第一（L466-469）：其从左到右扫描高效，提供更紧的初始上界，减少后续 group 工作量。
- 命中长度必须是各 group block_size 的 LCM 的倍数（不支持部分块命中）。
- EAGLE 每个候选长度最多 drop 一次（issue #32802）。

### 7.8 门面：KVCacheManager（`kv_cache_manager.py`）

Scheduler **只接触这一层**。`KVCacheManager`（[kv_cache_manager.py:106-542](../../vllm/v1/core/kv_cache_manager.py#L106-L542)）持有 coordinator 并暴露 `block_pool`（供 usage/events/evict）。

#### KVCacheBlocks

[kv_cache_manager.py:21-103](../../vllm/v1/core/kv_cache_manager.py#L21-L103) 是 Scheduler 与管理器之间的接口，隐藏内部结构。`blocks: tuple[Sequence[KVCacheBlock], ...]` 外层按 group 组织（`blocks[i][j]` 是第 i group 的第 j 块），而非按 token 块——当前所有 group 块数相同，但外层 group 化更稳健。预构造空对象（`empty_kv_cache_blocks`）避免 GC 开销。

#### `get_computed_blocks(request)`（[L183-223](../../vllm/v1/core/kv_cache_manager.py#L183-L223)）

前缀缓存查询入口：
1. 禁用缓存或 `skip_reading_prefix_cache` → 返回空；
2. **`max_cache_hit_length = request.num_tokens - 1`**——即使全部命中也必须重算最后一个 token 以获取 logits（可能导致整个块重算，未来可优化）；
3. `coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)`；
4. 记录 prefix cache stats（含 preempted 标记）；
5. 返回 `(KVCacheBlocks, num_new_computed_tokens)`。

#### `allocate_slots(...)`（[L225-416](../../vllm/v1/core/kv_cache_manager.py#L225-L416)）——核心方法

块布局（[L262-283](../../vllm/v1/core/kv_cache_manager.py#L262-L283)）：

```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
                                  | < to be computed >              |
                  | < to be allocated >                              |
```
- comp = `request.num_computed_tokens`（已有计算）
- new_comp = 前缀缓存新命中（`len(new_computed_blocks) * block_size`）
- ext_comp = connector 外部缓存
- new = 本步新 token（含未验证 draft）
- lookahead = 投机 token

三阶段（[L300-308](../../vllm/v1/core/kv_cache_manager.py#L300-L308)）：
1. 释放 comp 中不需要的块（如滑动窗口外），检查空闲块是否足够；
2. 处理前缀 token（comp + new_comp + ext_comp）：为 ext_comp 分配窗口内新块；
3. 为待计算 token（new + lookahead）分配新块。

详细流程：
1. `num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens`；
2. `total_computed_tokens = min(local + external, max_model_len)`；
3. **全序列准入检查**（`full_sequence_must_fit`，[L335-349](../../vllm/v1/core/kv_cache_manager.py#L335-L349)）：用 `apply_admission_cap=True` 估算完整序列所需块，超空闲则返回 None；
4. `num_tokens_need_slot = min(total_computed + num_new_tokens + lookahead, max_model_len)`；
5. **先 `remove_skipped_blocks`** 再估算需求（即使后续无法调度也值得做，减少驱逐）；
6. `get_num_blocks_to_allocate`（不带 cap）超空闲则返回 None；
7. `allocate_new_computed_blocks`（先追加命中块以防后续分配失败）；
8. `allocate_new_blocks` 得到新块；
9. 若 `not enable_caching or delay_cache_blocks` 直接返回（P/D 延迟缓存）；
10. **`num_tokens_to_cache = min(total_computed + num_new_tokens, request.num_tokens)`**——cap 到 `request.num_tokens` 确保只缓存"已定稿"token，**被拒的 draft token 不会污染前缀缓存**；
11. `coordinator.cache_blocks(request, num_tokens_to_cache)`；
12. 返回新块的 `KVCacheBlocks`。

> **`apply_admission_cap` 的二分语义**：仅在全序列准入检查时为 True；per-step 分配必须为 False 以精确匹配 `allocate_new_blocks` 的预测，否则会死锁或 OOM（[kv_cache_coordinator.py:105-108](../../vllm/v1/core/kv_cache_coordinator.py#L105-L108)）。

`allocate_slots` 返回 None 是调度器触发抢占/停止接纳的信号。

其他门面方法：`free(request)`（逆序释放）、`reset_prefix_cache()`、`get_num_common_prefix_blocks()`、`take_new_block_ids()`（收集需清零的新块）、`new_step_starts()`（委托，Mamba 清空本步缓存集合）、`evict_blocks()`（供 connector）。

### 7.9 块分配/释放/驱逐完整时序

**新请求到达（prefill）**：
1. Scheduler → `get_computed_blocks(request)` → `coordinator.find_longest_cache_hit` → 各 manager 查 `block_pool.get_cached_block`；
2. Scheduler → `allocate_slots(...)`：
   a. `remove_skipped_blocks`（SWA/Mamba 回收旧块）；
   b. 全序列准入 + per-step 容量检查；
   c. `allocate_new_computed_blocks`：`touch` 命中块（ref_cnt++，从 free queue 摘除），null 填 skipped 位；
   d. `allocate_new_blocks`：`block_pool.get_new_blocks` → `popleft_n` → 带 hash 的块先 `_maybe_evict_cached_block` → ref_cnt=1；
   e. `cache_blocks`：满块设 hash、插入哈希表。

**请求完成/抢占**：
1. `free(request)` → 各 manager `reversed(req_blocks)` → `block_pool.free_blocks`：
   - 所有块 ref_cnt -= 1；
   - ref_cnt 归 0 的非 null 块 append 回 free queue 尾部（LRU 最新端）；
   - 带 hash 的块留在哈希表中作为驱逐候选，直到被 popleft 时才驱逐。

**块驱逐（分配时隐式发生）**：
- `get_new_blocks` 从 free queue 头部（LRU 最旧端）popleft；
- 若块带 hash（cached 但 ref_cnt=0），`_maybe_evict_cached_block` 从哈希表移除并 reset_hash；
- 无 hash 的空闲块直接分配。

### 7.10 指标采集（`kv_cache_metrics.py`）

`KVCacheMetricsCollector`（[kv_cache_metrics.py:46-96](../../vllm/v1/core/kv_cache_metrics.py#L46-L96)）是可选观测组件，1% 采样率：
- `BlockMetricsState` 记录块的 birth_time、last_access、有界 access_history（maxlen=4）；
- `on_block_allocated/accessed/evicted` 回调计算 lifetime/idle/reuse_gaps；
- 通过 `metrics_collector` 参数注入，不传则零开销。

---

## 八、Encoder Cache Manager（`encoder_cache_manager.py`）

多模态模型的 encoder 输出（如视觉 embedding）需要独立缓存，与 decoder KV cache 平行。

### 8.1 设计定位

缓存粒度是**单个多模态输入项（image/audio item）**而非 encoder token（[encoder_cache_manager.py:18-65](../../vllm/v1/core/encoder_cache_manager.py#L18-L65)）——多模态 embedding 之间的 text/break token 不计入缓存容量。核心特性：
- 跨请求共享相同 `mm_hash` 的 embedding；
- 淘汰在 `can_allocate()` 分配时惰性发生；
- 优先淘汰无请求引用的最老条目。

### 8.2 核心数据结构（[L67-77](../../vllm/v1/core/encoder_cache_manager.py#L67-L77)）

| 字段 | 含义 |
|------|------|
| `cache_size` | 总容量（按 encoder embeddings 数量） |
| `num_free_slots` | 物理空闲槽位 |
| `num_freeable_slots` | 可立即回收的槽位（无引用条目占用） |
| `cached: dict[str, set[str]]` | `mm_hash → 引用该 embedding 的 request_id 集合`（空集表示存在但无人引用） |
| `freeable: OrderedDict[str, int]` | 可淘汰条目 `mm_hash → num_embeds`，按插入顺序 FIFO |
| `freed: list[str]` | 自上次 `get_freed_mm_hashes()` 以来被淘汰的 hash |

`cached` 与 `freeable` 是互补视图：hash 在 `cached` 中就物理存在；引用集变空时同时进入 `freeable` 等待淘汰。

### 8.3 两阶段回收（延迟物理释放）

- `free_encoder_input`（[L221-240](../../vllm/v1/core/encoder_cache_manager.py#L221-L240)）只把条目移入 `freeable`（引用集清空，`num_freeable_slots++`），**不立即释放物理内存，也不修改 `num_free_slots`**；
- `can_allocate`（[L119-178](../../vllm/v1/core/encoder_cache_manager.py#L119-L178)）空间不足时从 `freeable` 头部 FIFO `popitem(last=False)` 淘汰，从 `cached` 删除并记入 `freed`，`num_free_slots += num_embeds`；
- `get_freed_mm_hashes()`（[L255-266](../../vllm/v1/core/encoder_cache_manager.py#L255-L266)）取出 `freed` 列表，Scheduler 把它放入 `SchedulerOutput.free_encoder_mm_hashes` 通知 worker **物理释放**。

这种设计使"复活"（`check_and_update_cache` 命中 `freeable` 中的条目，从 freeable 摘除并重新引用）零成本。`allocate` 同时扣减 `num_free_slots` 和 `num_freeable_slots`（新分配既消耗物理空位，也消耗理论可回收容量）。

### 8.4 EncoderDecoderCacheManager

[encoder_cache_manager.py:323-381](../../vllm/v1/core/encoder_cache_manager.py#L323-L381) 是 encoder-decoder 模型（Whisper、T5）的简化临时实现：
- **不跨请求共享**（`check_and_update_cache` 永远返回 False）；
- 无淘汰，只检查 `num_free_slots`；
- 巧妙的**双缓冲**：`get_freed_mm_hashes` 交换 `allocated` 和 `to_free`，确保 entry 在模型执行**之后**才被 runner 释放（runner 在模型执行前释放 `to_free`），模拟父类的时序。

### 8.5 与 Scheduler 的交互

`_try_schedule_encoder_inputs`（[scheduler.py:1061-1222](../../vllm/v1/core/sched/scheduler.py#L1061-L1222)）决定哪些编码器输入本步处理，条件：
1. 编码器输出的 token 范围与本步计算范围重叠；
2. 尚未计算且不在编码器缓存中；
3. 远程编码器缓存也没有；
4. 有足够编码器 token 预算；
5. 编码器缓存有空间。

`disable_chunked_mm_input=True` 时，若只能部分覆盖某个 mm item，会把 `num_new_tokens` 回滚到该 mm item 之前，保证多模态输入不被分块。

---

## 九、完整调用时序

把调度器和 KV Cache 串起来，一个 step 的完整流程：

```
EngineCore.step()
│
├─ [1] scheduler.schedule()
│   │
│   ├─ kv_cache_manager.new_step_starts()         # Mamba 清空本步缓存集合
│   │
│   ├─ [阶段1] 遍历 running 请求:
│   │   ├─ 计算 num_new_tokens = num_tokens_with_spec + placeholders - computed
│   │   ├─ _try_schedule_encoder_inputs()         # 可能缩小 num_new_tokens
│   │   ├─ kv_cache_manager.allocate_slots()      # 分配 KV 块
│   │   │   ├─ coordinator.remove_skipped_blocks()  # SWA/Mamba 回收
│   │   │   ├─ 全序列准入检查 (apply_admission_cap=True)
│   │   │   ├─ coordinator.allocate_new_computed_blocks()  # touch 前缀命中块
│   │   │   ├─ block_pool.get_new_blocks()        # popleft + 驱逐
│   │   │   └─ coordinator.cache_blocks()         # 满块设 hash
│   │   ├─ 失败 → _preempt_request() (FCFS LIFO / PRIORITY 选最低优先级)
│   │   └─ 记录 num_scheduled_tokens, 扣预算
│   │
│   ├─ [阶段2] 无抢占时遍历 waiting/skipped:
│   │   ├─ kv_cache_manager.get_computed_blocks()  # 前缀缓存命中
│   │   ├─ connector.get_num_new_matched_tokens() # 外部 KV 命中
│   │   ├─ kv_cache_manager.allocate_slots(full_sequence_must_fit=True)
│   │   ├─ 加入 running，分类 new/resumed
│   │   └─ 失败则 break（waiting 不抢占）
│   │
│   ├─ 构造 NewRequestData / CachedRequestData
│   ├─ get_num_common_prefix_blocks()             # cascade attention
│   ├─ take_new_block_ids() → new_block_ids_to_zero
│   ├─ 构造 SchedulerOutput
│   └─ _update_after_schedule()                   # 推进 num_computed_tokens
│
├─ [2] model_executor.execute_model(scheduler_output)  # GPU 前向
├─ [3] scheduler.get_grammar_bitmask()            # 结构化输出约束
├─ [4] future.result() → ModelRunnerOutput
│
└─ [5] scheduler.update_from_output(scheduler_output, model_output)
    ├─ 投机解码接受/拒绝，回退 num_computed_tokens
    ├─ _update_request_with_output()
    │   ├─ request.append_output_token_ids()      # 追加 token + 更新 block_hashes
    │   └─ check_stop()                           # EOS/stop/length/repetition
    ├─ _handle_stopped_request() → _free_request()  # 释放 KV + encoder
    ├─ 移除 running/waiting 中停止的请求
    ├─ 收集 KV cache 事件并发布
    └─ 返回 EngineCoreOutputs（按 client_index 分组）
```

请求完整生命周期（从加入到清理）：

```
add_request() → WAITING 入 waiting 队列
   → schedule() 首次接纳: get_computed_blocks + allocate_slots → RUNNING
   → 多步 schedule/update_from_output 循环推进
      ├─ 资源不足 → _preempt_request() → PREEMPTED → waiting 队首 → 重新 RUNNING
      ├─ EOS/stop/length/repetition → FINISHED_*
      └─ 流式输入 → WAITING_FOR_STREAMING_REQ → 收到 update → WAITING
   → _free_request(): connector.request_finished + free encoder + free KV blocks
   → finished_req_ids 通知 worker 释放缓存 → 从 self.requests 删除
```

---

## 十、关键配置项

| 配置 | 来源 | 作用 |
|------|------|------|
| `max_num_seqs` | scheduler_config | 最大 running 请求数 |
| `max_num_batched_tokens` | scheduler_config | 每步 token 预算（`max_num_scheduled_tokens` 未设时回退） |
| `max_num_scheduled_tokens` | scheduler_config | 每步最大调度 token 数（显式覆盖） |
| `long_prefill_token_threshold` | scheduler_config | 长 prefill 分块阈值，0 不限 |
| `enable_chunked_prefill` | scheduler_config | 是否允许分块 prefill |
| `policy` | scheduler_config | `"fcfs"` 或 `"priority"` |
| `scheduler_reserve_full_isl` | scheduler_config | 准入时是否检查完整 ISL 能装入（默认 True） |
| `disable_chunked_mm_input` | scheduler_config | 是否禁止多模态输入分块 |
| `async_scheduling` | scheduler_config | 是否用 AsyncScheduler |
| `max_model_len` | model_config | 模型最大序列长度 |
| `num_gpu_blocks` | cache_config | GPU KV cache block 总数 |
| `block_size` | cache_config | KV cache block 大小 |
| `enable_prefix_caching` | cache_config | 是否启用前缀缓存 |
| `hash_block_size` | cache_config | 前缀缓存哈希块大小（默认 = block_size） |
| `mamba_cache_mode` | cache_config | Mamba 缓存模式（"align" 需 block 对齐） |
| `disable_hybrid_kv_cache_manager` | scheduler_config | True 时所有层按同尺寸分配（禁用混合优化） |
| `kv_load_failure_policy` | kv_transfer_config | `"recompute"` 或 `"fail"` |
| `max_loras` | lora_config | 最大并发 LoRA 数 |
| `num_speculative_tokens` | speculative_config | 投机解码 draft token 数 |
| `VLLM_USE_V2_MODEL_RUNNER` | envs | 是否用 v2 model runner（影响 preempted_req_ids 等字段） |

---

## 十一、关键设计决策总结

1. **统一 token 追赶模型**：无 prefill/decode 阶段之分，只有 `num_computed_tokens` 追赶 `num_tokens_with_spec`，天然统一 chunked prefill、prefix caching、spec decoding。

2. **RUNNING 优先 + 抢占时不接纳新请求**：先调度 running，仅无抢占时才接纳 waiting，防止被抢占请求饥饿。

3. **重算式抢占**：抢占即释放全部 KV、`num_computed_tokens` 归零、回 waiting 队首优先恢复。FCFS 用 LIFO 保护早到请求；PRIORITY 按 `(priority, arrival_time)` 选最低优先级者。

4. **物理块集中、块表分散、门面隐藏**：BlockPool 是物理块唯一所有者；每 group 维护独立 `req_to_blocks`；Scheduler 只接触 KVCacheManager 门面。

5. **引用计数 + 不可变块，无 COW**：共享只读块（前缀命中 touch 增引用）+ 新块追加，避免写时复制复杂性。ref_cnt 归 0 的块带 hash 留在哈希表中作为 LRU 驱逐候选。

6. **null_block（block 0）永久征用**：所有 manager 共享，用于滑动窗口/Mamba 块表中"无块"位置的占位对齐，永不缓存/释放/计入使用率。

7. **draft token 不污染前缀缓存**：`num_tokens_to_cache` cap 到 `request.num_tokens`（已定稿），被拒的 speculative token 不写入缓存哈希。

8. **延迟推进 computed tokens**：在 `schedule()` 末尾而非 `update_from_output` 推进，支持连续 chunked prefill；spec 拒绝时回退。

9. **异步调度的 placeholder 机制**：预留 `1 + num_spec_tokens` 个占位符，在 worker 生成实际 token 后才缓存 blocks，实现调度与执行解耦、消除 GPU 气泡。

10. **两阶段编码器缓存回收**：`free_encoder_input` 只移入 freeable（逻辑释放），淘汰时经 `free_encoder_mm_hashes` 通知 worker 物理释放，使复活零成本。

11. **逆序释放 + LRU**：`free` 时 `reversed(req_blocks)`，让块链尾部（复用价值最低）先进入空闲队列头部、先被驱逐。

12. **全序列准入闸门**：`scheduler_reserve_full_isl` 在 chunked prefill 下检查完整序列（而非仅首 chunk）能否装入，防止 KV cache 抖动；`apply_admission_cap` 仅此时为 True，per-step 必须 False。

13. **混合 group 的不动点算法**：HybridKVCacheCoordinator 迭代收敛不同 block_size group 的命中长度，LCM 对齐，FullAttention 优先扫描提供紧上界。

14. **零分配热路径**：FreeKVCacheBlockQueue 直接操作链表指针；BlockHashToBlockMap 单块时不创建内层 dict；空 KVCacheBlocks 预构造——这些都在尽力降低调度热路径的 GC 压力。

---

## 十二、与其他模块的关系

| 模块 | 交互方式 |
|------|----------|
| `vllm/v1/engine/` | EngineCore 每步调 `schedule()`、`update_from_output()`、`get_grammar_bitmask()`；初始化时创建 Scheduler 并传 KVCacheConfig |
| `vllm/v1/worker/` | 消费 `SchedulerOutput`：用 `block_ids` 构建 block table、`num_scheduled_tokens` 确定 input IDs、`finished_req_ids` 释放缓存、`new_block_ids_to_zero` 清零 |
| `vllm/v1/attention/` | `num_common_prefix_blocks` 驱动 cascade attention；各 attention backend 依赖 block table 布局 |
| `vllm/v1/spec_decode/` | 调度器维护 `spec_token_ids`、`num_lookahead_tokens`，处理 draft 接受/拒绝 |
| `vllm/v1/structured_output/` | `get_grammar_bitmask()` 生成约束位掩码；`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 状态 |
| `vllm/v1/kv_offload/`、KV connector | P/D 分离与 KV offload：`get_num_new_matched_tokens`、`update_state_after_alloc`、`request_finished`、`evict_blocks` |
| `vllm/config/` | `SchedulerConfig.get_scheduler_cls()` 选择 Scheduler/AsyncScheduler；`CacheConfig` 提供 block_size、num_blocks、prefix caching |
| `vllm/v1/request.py` | `Request` / `RequestStatus` 是调度器操作的核心对象 |
