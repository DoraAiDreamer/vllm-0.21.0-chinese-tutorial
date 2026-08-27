# 分布式推理

> 源码路径: `vllm/distributed/`

vLLM 的高性能很大程度来自把一个模型横向扩展到多卡/多节点。这个目录实现了多种并行策略与跨进程/跨节点的数据通路，对上层（模型层、Worker、调度器）基本透明——模型代码只写 `ColumnParallelLinear`/`RowParallelLinear`、调 `get_pp_group()` 收发中间张量，具体的 NCCL 通信、all-to-all、KV 传输都封装在这里。

本篇按"层"来组织：

1. **并行状态与通信组**（`parallel_state.py`）：TP/PP/DP/EP/CP 各组的 rank 布局、`GroupCoordinator`、集合通信原语；
2. **设备通信器**（`device_communicators/`）：NCCL wrapper、自定义 all-reduce、all-to-all 管理器（DeepEP/NiXL/FlashInfer）、共享内存广播；
3. **KV Transfer / KV Connector**（`kv_transfer/`）：PD 分离、KV cache offload 的可插拔连接器；
4. **EC Transfer**（`ec_transfer/`）：预填充/解码分离时的多模态编码器输出传输；
5. **EPLB**（`eplb/`）：专家并行负载均衡；
6. **Weight Transfer**（`weight_transfer/`）：在线/离线权重热更新；
7. **Elastic EP**（`elastic_ep/`）：专家并行弹性扩缩容。

> 前面篇章已经用到很多这里的概念：第 04 篇 Worker 的 `init_worker_distributed_environment`、PP 的 `isend/irecv_tensor_dict`；第 05 篇 Linear 层的 all-reduce/all-gather；第 06 篇 MoE 的 EP all-to-all 和 EPLB。本篇把它们串成完整图景。

---

## 整体结构

```
vllm/distributed/
├── parallel_state.py          # 核心：所有通信组、GroupCoordinator、init_*、集合通信
├── utils.py                   # 并行工具（rank 计算、PP send/recv buffer、partition 等）
├── communication_op.py        # 通信 op 注册（fused scaled matmul reduce-scatter 等）
├── kv_events.py               # KV cache 事件（前缀缓存命中/失效/存储）
├── nixl_utils.py              # NiXL（NVIDIA Xfer Library）工具
├── stateless_coordinator.py   # 无状态通信组（Elastic EP 用）
│
├── device_communicators/      # 底层通信实现
│   ├── base_device_communicator.py   # DeviceCommunicatorBase / All2AllManagerBase / Cache
│   ├── cuda_communicator.py          # CudaCommunicator
│   ├── pynccl.py / pynccl_wrapper.py / pynccl_allocator.py  # 直接 ctypes 调 NCCL
│   ├── cuda_wrapper.py               # CUDA runtime ctypes（IPC mem handle 等）
│   ├── all2all.py                    # 各 EP all-to-all 实现：
│   │     # AgRsAll2AllManager、DeepEPHT/LL、NixlEPAll2AllManager、
│   │     # FlashInferNVLink TwoSided/OneSided、MoriAll2AllManager
│   ├── custom_all_reduce.py          # CustomAllreduce（P2P 环形 all-reduce）
│   ├── flashinfer_all_reduce.py      # FlashInfer Sambo all-reduce
│   ├── quick_all_reduce.py           # QuickAllReduce
│   ├── all_reduce_utils.py
│   ├── symm_mem.py                   # 对称内存通信器
│   ├── shm_broadcast.py / shm_object_storage.py  # 同机共享内存广播/对象存储
│   ├── cpu_communicator.py / xpu_communicator.py / ray_communicator.py
│   └── mnnvl_compat.py
│
├── kv_transfer/               # KV cache 传输（PD 分离 / offload）
│   ├── kv_transfer_state.py   # get_kv_transfer_group / has_kv_transfer_group
│   └── kv_connector/
│       ├── base.py / factory.py / utils.py
│       └── v1/
│           ├── base.py                # KVConnectorBase_V1（核心抽象）
│           ├── metrics.py
│           ├── nixl/                  # NiXL 后端 connector（worker/scheduler/metadata/tp_mapping）
│           ├── mooncake/              # Mooncake connector
│           ├── lmcache/               # LMCache connector（含多进程 mp）
│           ├── offloading/            # 通用 offloading（scheduler/worker/common）
│           ├── moriio/                # MoriIO
│           ├── flexkv_connector.py / simple_cpu_offload_connector.py
│           ├── example_connector.py / example_hidden_states_connector.py
│           ├── multi_connector.py / decode_bench_connector.py
│           └── ssm_conv_transfer_utils.py
│
├── ec_transfer/               # Encoder Compute transfer（PD 分离的编码器部分）
│   ├── ec_transfer_state.py   # get_ec_transfer / has_ec_transfer
│   └── ec_connector/
│       ├── base.py            # ECConnectorBase（producer/consumer）
│       ├── factory.py
│       └── example_connector.py
│
├── eplb/                      # Expert-Parallel Load Balancing
│   ├── eplb_state.py          # EplbState/EplbModelState/EplbLayerState/EplbStats
│   ├── eplb_communicator.py   # 多种后端（TorchDist NCCL/Gloo staged、NiXL、PyNccl）
│   ├── rebalance_execute.py   # 执行专家重平衡（拷贝专家权重）
│   ├── async_worker.py
│   └── policy/                # 重平衡策略
│
├── weight_transfer/           # 训练→推理在线权重更新
│   ├── base.py                # WeightTransferEngine 抽象
│   ├── factory.py
│   ├── nccl_engine.py         # NCCL 广播权重
│   ├── ipc_engine.py          # 同机 CUDA IPC 传输
│   └── packed_tensor.py
│
└── elastic_ep/                # 弹性专家并行（运行时增删 EP rank）
    ├── elastic_execute.py
    ├── elastic_state.py
    └── standby_state.py
```

---

## 一、并行维度与 rank 布局

vLLM 同时支持多种并行，它们在 rank 空间里是**正交嵌套**的。[parallel_state.py:1560](../../vllm/distributed/parallel_state.py#L1560) 的注释明确了布局顺序：

```
ExternalDP x DP x PP x PCP x TP
```

- **TP（Tensor Parallel）**：切单层权重，`ColumnParallelLinear`/`RowParallelLinear` 配合 all-reduce/all-gather；同一 TP 组内 rank 通常在同一节点（NVLink）。
- **PP（Pipeline Parallel）**：按层切分模型，不同 rank 持有不同层区间，靠 P2P send/recv 传 `IntermediateTensors`。
- **PCP/DCP（Prefill/Decode Context Parallel）**：把长上下文的 KV/attention 计算切到多 rank；PCP 用于 prefill，DCP 用于 decode（DCP 复用 TP 的 GPU，把一个 TP 组再细分）。
- **DP（Data Parallel）**：复制模型、各自处理不同请求；同一 DP 组必须**同步**调用 `generate`，否则死锁。
- **ExternalDP**：独立于模型的数据并行（如 verl 集成），各 DP rank 可独立生成。
- **EP（Expert Parallel）**：MoE 专家分布到不同 rank，token 经 all-to-all 路由到对应专家。EP 组默认与 TP/DP 组重合，但也可独立配置。

### 1.1 初始化顺序

Worker 的 `init_worker_distributed_environment`（第 04 篇）依次调用：

```
init_distributed_environment(...)          # 建 torch 的 WORLD 进程组（NCCL）
   └─ init_world_group(...)              # _WORLD = GroupCoordinator(所有 rank)
ensure_model_parallel_initialized(tp, pp, pcp, dcp)
   └─ initialize_model_parallel(...)      # 建 TP/PP/PCP/DCP/DP/EP/EPLB 各组
```

[init_distributed_environment](../../vllm/distributed/parallel_state.py#L1358) 先处理 DP rank 偏移（多节点或 DP 时按 `data_parallel_rank * world_size + rank` 调整全局 rank/world_size，并选独立的 DP master ip/port），再 `torch.distributed.init_process_group(backend="nccl")`，最后构造 `_WORLD`。

[initialize_model_parallel](../../vllm/distributed/parallel_state.py#L1494) 把全局 `arange(world_size)` reshape 成 `(-1, DP, PP, PCP, TP)`，然后对每个维度 `transpose + unbind` 得到各组的 rank 列表，逐个调 `init_model_parallel_group(group_ranks, ..., group_name=...)` 建立：

| 全局变量 | 组 | 典型通信 |
|----------|----|---------|
| `_WORLD` | 所有 rank | 广播、全局 barrier |
| `_TP` | 同层切权重的 rank | all-reduce（RowParallel 输出）、all-gather（ColumnParallel 输出）、MoE dispatch/combine |
| `_PP` | 持有不同层的 rank | P2P send/recv 中间张量 |
| `_PCP`/`_DCP` | 上下文并行 rank | KV/attention 的 ring/halo 交换 |
| `_DP` | 数据并行副本 | 批量协调（外部 launcher） |
| `_EP` | 专家并行 rank | all-to-all 路由 token |
| `_EPLB` | EPLB 重平衡通信 | 专家权重重分布 |

访问器 `get_tp_group()`/`get_pp_group()`/`get_ep_group()`/...（[L1229-1288](../../vllm/distributed/parallel_state.py#L1229)）返回对应 `GroupCoordinator`。`ensure_model_parallel_initialized`（[L1738](../../vllm/distributed/parallel_state.py#L1738)）保证幂等，避免重复初始化。

---

## 二、GroupCoordinator：通信组的统一封装

[parallel_state.py:290](../../vllm/distributed/parallel_state.py#L290) 的 `GroupCoordinator` 是所有进程组的包装，屏蔽了单卡、NCCL、Ray、自定义通信器等差异。

### 2.1 状态与设备组

它持有：
- `ranks`、`rank_in_group`、`world_size`；
- `device_group`：torch 的 NCCL 进程组（GPU 张量走它）；
- `cpu_group`：gloo 进程组（元数据/对象走它，避免 GPU 同步）；
- `device_communicator`：可选的自定义高速通信器（见第三节）；
- `use_message_queue_broadcaster`：TP 组用消息队列广播器做低延迟控制面广播。

便捷属性 `is_first_rank/is_last_rank/first_rank/last_rank/next_rank/prev_rank`（[L430-463](../../vllm/distributed/parallel_state.py#L430)）让 PP 代码很简洁。

### 2.2 集合通信

标准集合算子（[L502-606](../../vllm/distributed/parallel_state.py#L502)）：
- `all_reduce(input_)` → `tensor_model_parallel_all_reduce`；
- `all_gather(input_, dim)` / `_out_place` 变体；
- `reduce_scatter(input_, dim)`；
- `gather/broadcast/send/recv/barrier`；
- 对 CPU 张量自动用 cpu_group。

`set_custom_all_reduce(enable)`（[L1318](../../vllm/distributed/parallel_state.py#L1318)）启用后，TP 组的 `all_reduce` 会走自定义 P2P 实现（第三节）。

### 2.3 张量字典收发（PP 核心）

第 04 篇 Worker 用的 `send_tensor_dict`/`isend_tensor_dict`/`recv_tensor_dict`/`irecv_tensor_dict`（[L821-1040](../../vllm/distributed/parallel_state.py#L821)）是 PP 的数据通路：

- 把 `dict[str, tensor]` 拆成 **metadata（键名/形状/dtype）+ tensor 列表**；
- metadata 通过 `cpu_group` 用 `send_object/recv_object` 传（小、可序列化）；
- 每个 tensor 用 `torch.distributed.isend/irecv` 在 `device_group` 上**非阻塞** P2P 传输，返回 `Handle`；
- 接收方先用 metadata 分配张量，再启动 irecv，返回 `(tensor_dict, handles, postprocess)`；调用方等所有 handle 完成后跑 postprocess。
- **all-gather 优化**（[L807](../../vllm/distributed/parallel_state.py#L807) `_should_use_all_gather`）：当接收方同时需要跨 TP 聚合一个张量时（如 SP 场景），发送方把张量 reshape 成 `(tp_size, -1)` 只发自己那一片，接收方用 all-gather 重组，减少跨 stage 传输量。`all_gather_tensors` 参数允许逐键指定。
- 可选 `use_cpu_custom_send_recv` 走共享内存自定义通路。

返回的 `Handle`（[L73](../../vllm/distributed/parallel_state.py#L73)）是带 `is_completed/wait` 的协议，第 04 篇 `AsyncIntermediateTensors` 就用它做懒同步。

### 2.4 graph_capture 上下文

[graph_capture(device)`（[L1294](../../vllm/distributed/parallel_state.py#L1294)）在 CUDA graph 捕获期间，把 `GroupCoordinator` 切换到基于 **CUDA graph-safe 的 P2P buffer** 模式（`monkeypatch_P2P`），保证捕获时通信也被录进图里。

---

## 三、设备通信器

[device_communicators/](../../vllm/distributed/device_communicators/) 提供比 torch.distributed 更快或更专用的通信实现。

### 3.1 PyNccl：ctypes 直调 NCCL

- [pynccl_wrapper.py](../../vllm/distributed/device_communicators/pynccl_wrapper.py) 用 ctypes 定义 `ncclUniqueId`、数据类型、归约 op 枚举和 `NCCLLibrary`（`ncclCommInitRank/AllReduce/AllGather/ReduceScatter/Send/Recv/GroupStart/GroupEnd/...`）。
- [pynccl.py](../../vllm/distributed/device_communicators/pynccl.py) 的 `PyNcclCommunicator` 封装一个 NCCL 通信器，提供 `all_reduce/all_gather/send/recv/cuda_graph_support` 等；它绕过 torch.distributed 的 Python 开销，被自定义 all-reduce、EPLB、weight transfer 等使用。
- [pynccl_allocator.py](../../vllm/distributed/device_communicators/pynccl_allocator.py) 管理 NCCL 对称内存（symmetric memory），用于单边通信。

### 3.2 自定义 All-Reduce

LLM 推理时 batch 小，NCCL all-reduce 的 kernel launch 开销显著。vLLM 提供多套 P2P 环形 all-reduce：

- **`CustomAllreduce`**（[custom_all_reduce.py:50](../../vllm/distributed/device_communicators/custom_all_reduce.py#L50)）：经典实现，rank 间通过 CUDA IPC 交换指针，在一个自定义 kernel 里做流水环归约，只对小尺寸（默认 ≤ 64MiB）启用，大尺寸回退 NCCL。
- **`FlashInferAllReduce`**（[flashinfer_all_reduce.py:229](../../vllm/distributed/device_communicators/flashinfer_all_reduce.py#L229)）：FlashInfer 的 Sambo all-reduce。
- **`QuickAllReduce`**（[quick_all_reduce.py:45](../../vllm/distributed/device_communicators/quick_all_reduce.py#L45)）：按 regime（消息大小）选择最优核。

`all_reduce_utils.py` 共享注册/选择逻辑。`GroupCoordinator.all_reduce` 在 `disable_custom_all_reduce=False` 时优先用这些。

### 3.3 All-to-All（EP 路由）

[all2all.py](../../vllm/distributed/device_communicators/all2all.py) 是 MoE expert parallel 的关键。`All2AllManagerBase`（[L30](../../vllm/distributed/device_communicators/all2all.py#L30)）抽象"把 token 按目标专家发到对应 rank、再收回来"的 dispatch/combine。具体实现：

| 类 | 后端 |
|----|------|
| `AgRsAll2AllManager`（[L40](../../vllm/distributed/device_communicators/all2all.py#L40)） | 经典 all-gather + reduce-scatter 组合（无第三方库） |
| `DeepEPHTAll2AllManager` / `DeepEPLL`（[L196/257](../../vllm/distributed/device_communicators/all2all.py#L196)） | DeepEP 高吞吐/低延迟 kernel |
| `NixlEPAll2AllManager`（[L327](../../vllm/distributed/device_communicators/all2all.py#L327)） | NiXL |
| `FlashInferNVLinkTwoSided/OneSided`（[L442/549](../../vllm/distributed/device_communicators/all2all.py#L442)） | FlashInfer NVLink |
| `MoriAll2AllManager`（[L671](../../vllm/distributed/device_communicators/all2all.py#L671)） | Mori |

`dispatch(d...)` 发送、`combine(...)` 收回并加权，FusedMoE runner 据此实现 token 路由。

### 3.4 共享内存通信

- [shm_broadcast.py](../../vllm/distributed/device_communicators/shm_broadcast.py)：同机多进程用的环形缓冲广播（带 `SpinCondition` 自旋），TP 组的消息队列广播器基于此。
- [shm_object_storage.py](../../vllm/distributed/device_communicators/shm_object_storage.py)：单写者共享内存对象存储，用 msgpack 序列化，用于跨进程传小对象（如多模态缓存，见第 04 篇 `mm_receiver_cache`）。
- [symm_mem.py](../../vllm/distributed/device_communicators/symm_mem.py)：对称内存通信器。

### 3.5 其他通信器

`CudaCommunicator`/`CpuCommunicator`/`XpuCommunicator` 继承 `DeviceCommunicatorBase`；`RayPPCommunicator` 在 Ray 后端用 Ray 的对象通道做 PP 通信（无 NCCL P2P 时）。[cuda_wrapper.py](../../vllm/distributed/device_communicators/cuda_wrapper.py) 通过 ctypes 调 `cuMemExportToShareableHandle`/`cuMemImportFromShareableHandle` 等 CUDA Driver API，用于跨进程 IPC。

---

## 四、KV Transfer：PD 分离与 KV Offload

[kv_transfer/](../../vllm/distributed/kv_transfer/) 实现 KV cache 在实例/层级间的搬运，最典型的两个场景：

- **Prefill-Decode (PD) 分离**：专门的 prefill 实例算完 prompt 的 KV，通过网络传给 decode 实例；
- **KV cache offload/分层缓存**：把不活跃的 KV 卸到 CPU/SSD/远端缓存（LMCache、Mooncake、FlexKV 等），需要时再加载。

核心是可插拔的 **KV Connector** 抽象。

### 4.1 全局状态与初始化

[kv_transfer_state.py](../../vllm/distributed/kv_transfer/kv_transfer_state.py) 提供 `get_kv_transfer_group()`/`has_kv_transfer_group()`/`ensure_kv_transfer_initialized(vllm_config, kv_cache_config)`。Worker 在 `initialize_from_config` 里（第 04 篇）调用后者：用 factory 按配置创建 connector 实例，注册 KV caches。

### 4.2 KVConnectorBase_V1

[kv_connector/v1/base.py:170](../../vllm/distributed/kv_transfer/kv_connector/v1/base.py#L170) 是 connector 的核心抽象。一个 connector 在**两侧**都有角色（`KVConnectorRole`，SCHEDULER/WORKER 或两者），关键方法按调度/执行阶段排列：

**Scheduler 侧（决定加载/保存什么）：**
- `get_num_new_matched_tokens(...)`：根据请求前缀在外部 KV 库里查命中了多少 token（影响调度器的 `num_computed_tokens`，启用 prefix caching/PD 命中）；
- `update_state_after_alloc(...)`：分配块后更新状态；
- `build_connector_meta(...)`：构造要发给 worker 的每步元数据（哪些块要 load/save、远端句柄等）；
- `request_finished(...)`：请求结束清理。

**Worker 侧（实际搬运 KV）：**
- `register_kv_caches(kv_caches)` / `register_cross_layers_kv_cache(...)`：绑定 model runner 分配的 KV cache 张量；
- `start_load_kv(forward_context, **kwargs)`：**前向前**启动异步 KV 加载（把远端/CPU KV 拷进本层的 KV cache）；
- `wait_for_layer_load(layer_name)`：某层计算前等待该层的 KV 加载完成（细粒度重叠——加载第 N 层时第 N-1 层已经在算）；
- `save_kv_layer(layer_name, kv_cache, attn_metadata)`：前向后/中保存某层的 KV；
- `wait_for_save()`：等所有保存完成；
- `get_finished()`/`get_block_ids_with_load_errors()`：汇报完成与加载失败块（失败块会被重新计算）。

其他：`get_handshake_metadata`（worker 间交换连接信息，如 NCCL unique id/NiXL 地址）、`get_kv_connector_stats`、`take_events`（KVCacheEvent 流）、`get_required_kvcache_layout`（要求 NCHW 等布局）、`requires_piecewise_for_cudagraph`（与编译图协作）。

这套接口设计的要点是**与 v1 的分层前向/异步流水线深度整合**：load 是逐层 wait、save 可在前向中穿插，不阻塞主计算流。

### 4.3 内置 connector

| Connector | 用途 |
|-----------|------|
| **nixl/**（[connector.py](../../vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py)） | 用 NVIDIA Xfer Library (NiXL) 做高速 PD 分离 KV 传输；含 scheduler/worker/tp_mapping/metadata/stats |
| **mooncake/** | Mooncake 传输引擎（Moonshot AI 的 KV 池化） |
| **lmcache/**（含 `lmcache_mp_connector`） | LMCache 分层缓存（CPU/磁盘/远端），支持多进程 |
| **offloading/** | 通用 KV offload 框架（scheduler + worker + common） |
| **moriio/** | MoriIO 引擎 |
| **flexkv_connector.py** | FlexKV |
| **simple_cpu_offload_connector.py** | 简单 CPU 卸示例/实现 |
| **multi_connector.py** | 组合多个 connector（如同时 offload + PD） |
| **example_connector.py / example_hidden_states_connector.py** | 开发者模板 |
| **decode_bench_connector.py** | 解码性能基准 |

[factory.py](../../vllm/distributed/kv_transfer/kv_connector/factory.py) 按配置的字符串/类名反射创建，并支持第三方插件注册。

### 4.4 KV 事件

[kv_events.py](../../vllm/distributed/kv_events.py)（527 行）定义 `KVCacheEvent` 体系：块存储、移除、命中、清空等事件流。connector 可以 `take_events()` 订阅这些事件以增量维护外部缓存，而不必每步全量轮询。

---

## 五、EC Transfer：编码器输出传输

[ec_transfer/](../../vllm/distributed/ec_transfer/) 处理**Encoder Compute (EC) 分离**——多模态模型中，prefill 实例（producer）跑完视觉编码器后，把编码器输出传给 decode 实例（consumer），避免在 decode 侧重复编码。

- [ec_transfer_state.py](../../vllm/distributed/ec_transfer/ec_transfer_state.py)：`get_ec_transfer()`/`has_ec_transfer()`/`ensure_ec_transfer_initialized(...)`。
- [ec_connector/base.py:59](../../vllm/distributed/ec_transfer/ec_connector/base.py#L59) 的 `ECConnectorBase` 与 KV connector 类似但更简单：producer 调 `save_caches(...)` 发送，consumer 调 `start_load_caches(...)`/`get_finished(...)` 接收；用 `mm_hash` 标识编码器输出。角色由 `ECConnectorRole` 区分，`is_producer/is_consumer` 判定。
- 第 04 篇 model runner 里 `maybe_get_ec_connector_output` 上下文、`_execute_mm_encoder` 在 producer 侧提前返回空输出，就是与它协作。
- 与 KV connector 的区别：EC 传的是编码器输出（一次性、按 mm_hash 缓存），不是逐层 KV。

---

## 六、EPLB：专家并行负载均衡

[eplb/](../../vllm/distributed/eplb/) 解决 MoE 专家负载不均（某些热门专家被更多 token 选中，导致对应 GPU 成为瓶颈）。EPLB **运行时动态迁移专家副本/重排专家放置**。

- **[eplb_state.py:210](../../vllm/distributed/eplb/eplb_state.py#L210) `EplbState`**：每个 MoE 模型一份（key 是 config hash）。
  - `EplbModelState`/`EplbLayerState`：记录每个专家的负载统计（滑动窗口 `expert_load_window`）、当前物理放置、冗余专家；
  - 用一个**共享的 GPU bool 标量** `should_record_tensor` 同时控制所有层是否记录本步负载（[L248](../../vllm/distributed/eplb/eplb_state.py#L248)）；
  - `expert_rearrangement_step` 计数，超过 `interval` 就触发重平衡（**所有 EP rank 必须同步**，否则集合通信挂死，[L229 注释](../../vllm/distributed/eplb/eplb_state.py#L229)）；
  - 支持 `is_async` 后台线程执行，不阻塞推理。
- **`policy/`**：决定如何重排专家（哪些专家复制/迁移）的算法。
- **[rebalance_execute.py](../../vllm/distributed/eplb/rebalance_execute.py)**：`TransferMetadata`/`AsyncEplbLayerResult`，实际在 rank 间拷贝专家权重、更新 `expert_map`（与第 06 篇 `update_physical_experts_metadata`、第 05 篇 FusedMoE 的 `expert_map` 联动）。
- **[eplb_communicator.py](../../vllm/distributed/eplb/eplb_communicator.py)**：多种重平衡通信后端：`TorchDistNcclEplbCommunicator`、`TorchDistGlooStagedEplbCommunicator`、`NixlEplbCommunicator`、`PyNcclEplbCommunicator`（[L44 起](../../vllm/distributed/eplb/eplb_communicator.py#L44)）。
- **`async_worker.py`**：异步执行循环，后台搬运权重。

第 04 篇 Worker 的 `eplb_step/is_dummy` 每步驱动 EPLB，模型 runner 里 `set_eplb_state` 把状态注入 FusedMoE 层。

---

## 七、Weight Transfer：在线权重更新

[weight_transfer/](../../vllm/distributed/weight_transfer/) 支持**训练进程与推理进程之间的在线权重同步**（如强化学习里 rollout 用最新策略权重）。

- **[base.py:49](../../vllm/distributed/weight_transfer/base.py#L49) `WeightTransferEngine`**（泛型 `[TInitInfo, TUpdateInfo]`）：
  - `init_transfer_engine(init_info)`：建立与训练端的连接/进程组；
  - `receive_weights(update_info, load_weights)`：接收一批权重并通过回调写入模型；
  - `parse_init_info/parse_update_info`：把 dict 解析成类型化 dataclass；
  - `trainer_send_weights(...)`：训练端发送；
  - `shutdown()`。
- **[nccl_engine.py](../../vllm/distributed/weight_transfer/nccl_engine.py)**：通过 NCCL（独立进程组）广播权重，支持 packed 传输减少 kernel 数。
- **[ipc_engine.py](../../vllm/distributed/weight_transfer/ipc_engine.py)**：同机用 CUDA IPC handle 直接共享/拷贝权重，零网络开销。
- **[factory.py:19](../../vllm/distributed/weight_transfer/factory.py#L19) `WeightTransferEngineFactory`**：按 `weight_transfer_config` 创建引擎（注册机制）。
- **[packed_tensor.py](../../vllm/distributed/weight_transfer/packed_tensor.py)**：把多个小张量打包成连续大张量传输。

第 04 篇 Worker 的 `init_weight_transfer_engine/start_weight_update/update_weights/finish_weight_update` 就是这套引擎的上层封装，配合 `model.load_weights` 支持 checkpoint 格式（逐层后处理）和 kernel 格式（直接 copy）。

---

## 八、Elastic EP：弹性专家并行

[elastic_ep/](../../vllm/distributed/elastic_ep/) 支持运行时**动态增减 EP worker**（专家数随负载扩缩），是较新的实验特性：

- [elastic_state.py](../../vllm/distributed/elastic_ep/elastic_state.py)：维护当前 EP 拓扑、生命周期状态；
- [standby_state.py](../../vllm/distributed/elastic_ep/standby_state.py)：待命 rank 的状态；
- [elastic_execute.py](../../vllm/distributed/elastic_ep/elastic_execute.py)：`ElasticEPScalingExecutor` 包装 Worker，在扩缩容时重路由 collective 调用、重建通信组。

它使用 [stateless_coordinator.py](../../vllm/distributed/stateless_coordinator.py) 的 `StatelessGroupCoordinator`——不依赖固定的 torch 进程组，可在 rank 变化时重建，配合 TCP store 协调。`init_distributed_environment` 里 `enable_elastic_ep` 分支（[L1439-1462](../../vllm/distributed/parallel_state.py#L1439)）走这条路径。

---

## 九、并行工具与上层协作

- **[utils.py](../../vllm/distributed/utils.py)**：`get_pp_indices`（按 rank 算层区间，第 06 篇 `make_layers` 用）、`split_tensor_along_last_dim`、`get_distributed_init_method`、张量布局工具等。
- **[communication_op.py](../../vllm/distributed/communication_op.py)**：注册可被编译/融合的通信算子（如 `fused_scaled_matmul_reduce_scatter`，序列并行/SP 用）。
- **[nixl_utils.py](../../vllm/distributed/nixl_utils.py)**：NiXL 初始化与元数据交换辅助。
- 与编译的协作：`graph_capture` 上下文让通信在 CUDA graph 内可重放；pass 里的 `allreduce_rms_fusion`、`collective_fusion`、`sequence_parallelism`（第 07 篇）把集合通信和相邻算子融合。

---

## 十、典型数据流回顾

### Tensor Parallel（一个 transformer block 内）

```
ColumnParallelLinear(qkv_proj)          RowParallelLinear(o_proj)
  每 rank 持部分权重                       每 rank 持部分权重
  X @ A_i  →  可选 all-gather 拼回完整 Q   分片输入 X_i @ B_i
  （attention 通常不 gather，直接用分片头）  → all-reduce 求和 → 完整输出
```
MoE 的 ColumnParallel(gate_up)/RowParallel(down) 同理；EP 时额外在专家前做 all-to-all dispatch、之后 combine。

### Pipeline Parallel（跨 rank）

```
stage i-1:  output = model(...)  →  IntermediateTensors{hidden,residual}
            pp_group.isend_tensor_dict(output)  ──P2P──┐
                                                         ▼
stage i:    pp_group.irecv_tensor_dict() → AsyncIntermediateTensors (懒 wait)
            model(..., intermediate_tensors) → ...
```
非阻塞 send 存在 Worker._pp_send_work 里，下一 step 先 wait（第 04 篇）。

### Prefill-Decode 分离（KV connector）

```
Prefill 实例:
  scheduler: get_num_new_matched_tokens=0; build_connector_meta（标记保存）
  worker: 每层前向 → save_kv_layer → wait_for_save → 发送 KV 块
Decode 实例:
  scheduler: get_num_new_matched_tokens=N（命中 prompt）→ num_computed_tokens=N
  worker: start_load_kv → 逐层 wait_for_layer_load → 注意力直接用远端 KV → 继续 decode
```

### EPLB 重平衡

```
每步: 记录专家负载到滑动窗口
每 interval: policy 算出新放置 → rebalance_execute 在 EP rank 间拷权重
           → 更新 FusedMoE.expert_map → 后续路由自动用新放置
（async 模式在后台线程搬运，is_async=True）
```

---

## 小结

1. **`parallel_state.py` 是中枢**：用一个 `(ExternalDP, DP, PP, PCP, TP)` 的多维 rank 网格构建所有通信组，`GroupCoordinator` 统一封装集合通信、PP 的非阻塞张量字典收发、CUDA graph 兼容；
2. **设备通信器提供性能与专用通路**：PyNccl 直调 NCCL、自定义 P2P all-reduce 降低小消息开销、多套 all-to-all 后端支撑 EP、共享内存广播服务控制面；
3. **KV Connector 是 PD 分离/分层缓存的可插拔抽象**，接口与 v1 分层前向的异步流水线紧耦合（逐层 load wait、逐层 save、事件驱动），NiXL/Mooncake/LMCache/offloading 等都是它的实现；
4. **EC Connector** 专门搬运多模态编码器输出；
5. **EPLB** 用滑动窗口统计 + 策略 + 跨 rank 权重重分布，在运行时均衡 MoE 专家负载；**Weight Transfer** 用 NCCL/IPC 支持训练-推理在线权重同步；**Elastic EP** 支持专家并行的弹性扩缩；
6. 所有这些对模型代码几乎透明——模型只看到并行 Linear、Attention、MoE 和 `get_pp_group()`，复杂性都收敛在本目录与 Worker/编译管线的协作里。

下一篇 [09 LoRA 与适配器](./09_lora.md) 将进入 `vllm/lora/`，看运行时热加载 LoRA、按批映射、与并行 Linear/Embedding/Conv 的集成。
