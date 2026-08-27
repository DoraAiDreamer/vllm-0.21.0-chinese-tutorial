# Worker 与模型运行器

> 源码路径: `vllm/v1/worker/`

Worker（工作进程）是 vLLM 真正"在 GPU 上干活"的一层。引擎核心（engine core）负责调度，它通过 RPC 把每个 step 的 `SchedulerOutput` 发给 Worker；Worker 再把这份调度结果翻译成模型能吃的 GPU 张量、跑一遍前向、做采样/池化，最后把 `ModelRunnerOutput` 还回去。

如果说前几篇梳理的是"大脑"（引擎、调度器、注意力后端），那这一篇就是"躯干与四肢"：

- **`Worker` / `WorkerWrapperBase`**：进程内的设备代理，管生命周期（初始化设备、加载权重、profile 显存、分配 KV cache、捕获 CUDA graph、执行）；
- **`GPUModelRunner`**（[gpu_model_runner.py](../../vllm/v1/worker/gpu_model_runner.py)，约 7200 行）：单卡上最核心的执行器，持有模型、KV cache、持久化输入张量、注意力元数据构造器、采样器、投机解码 drafter 等所有运行时状态；
- **`InputBatch` / `BlockTable`**：常驻显存/内存的"持久化批"数据结构，让连续 step 之间尽量只增删变化的请求；
- **一堆子模块**：采样（`gpu/sample/`）、多模态编码器（`gpu/mm/`）、池化（`gpu/pool/`）、CUDA graph、投机解码、kernel warmup、工作区（workspace）等。

> 说明：仓库里同时存在两套 model runner。默认（V1）是单文件的 `gpu_model_runner.py`；新版 V2 位于 `gpu/model_runner.py`，通过环境变量 `VLLM_USE_V2_MODEL_RUNNER` 开启（见 [gpu_worker.py:156](../../vllm/v1/worker/gpu_worker.py#L156)、[316-330](../../vllm/v1/worker/gpu_worker.py#L316-L330)）。V2 把逻辑拆成了 `InputBatch`/`RequestState`/`Sampler`（`gpu/` 目录）等更小组件，结构更清晰但思路一致。本文以默认 V1 为主，在关键处标注 V2 的对应拆分。

---

## 整体结构

```
vllm/v1/worker/
├── worker_base.py            # WorkerBase（设备无关接口）+ WorkerWrapperBase（进程内包装器）
├── gpu_worker.py             # Worker(WorkerBase)：GPU 实现，整个生命周期的编排者
├── gpu_model_runner.py       # GPUModelRunner：默认 V1 运行器（~7200 行，核心中的核心）
├── gpu_input_batch.py        # InputBatch + CachedRequestState：持久化批与请求缓存状态
├── block_table.py            # BlockTable / MultiGroupBlockTable：逻辑块→物理块表 + slot mapping
├── gpu_ubatch_wrapper.py     # UBatchWrapper：微批（DBO）模型包装
├── ubatching.py              # 微批上下文/流切换（compute/comm 重叠）
├── ubatch_utils.py           # 微批切片、注意力元数据切分
├── workspace.py              # 全局工作区张量管理器（flash-attn 等的临时 workspace）
├── utils.py                  # AttentionGroup、KV cache 分配/绑定、block size 协商、显存零块
├── mamba_utils.py / cp_utils.py / dp_utils.py / pp_utils  # SSM/上下文并行/数据并行/流水并行辅助
├── lora_model_runner_mixin.py        # LoRA 热加载/激活 mixin
├── kv_connector_model_runner_mixin.py # KV connector（PD 分离、跨层统一 KV cache）mixin
├── ec_connector_model_runner_mixin.py # Encoder connector（预填充/解码分离的编码器传输）mixin
├── encoder_cudagraph.py / encoder_cudagraph_defs.py  # 视觉编码器独立 CUDA graph
├── cpu_worker.py / cpu_model_runner.py   # CPU 后端
├── xpu_worker.py / xpu_model_runner.py   # XPU 后端
└── gpu/                      # 组件化实现（V2 runner 与 V1 共用的部分子模块）
    ├── model_runner.py       # V2 GPUModelRunner
    ├── input_batch.py        # V2 InputBuffers/InputBatch + 一堆 Triton 更新 kernel
    ├── states.py             # V2 RequestState
    ├── block_table.py        # V2 BlockTables（多组、指针表、gather/slot kernel）
    ├── buffer_utils.py       #  pinned host 缓冲（异步 H2D）
    ├── async_utils.py        # 异步输出（非阻塞 D2H 拷贝）
    ├── cudagraph_utils.py    # CUDA graph 辅助
    ├── warmup.py             # warmup_kernels()：两次假 prefill/decode 触发 Triton JIT
    ├── model_states/         # 模型专属状态接口（default/mamba_hybrid/whisper）
    ├── sample/               # V2 采样器与各 penalty/logprob/bad_words 状态
    ├── spec_decode/          # 投机解码：eagle、rejection sampler、ngram
    ├── mm/                   # 多模态编码器运行器与缓存
    ├── pool/                 # 池化模型运行器（含 late-interaction）
    ├── metrics/              # logits NaN 计数等
    └── structured_outputs.py # 结构化输出（grammar bitmask）
```

调用分层：

```
EngineCore (driver 进程)
   │  SchedulerOutput  ──collective_rpc("execute_model")──▶
   ▼
Executor (uni / mp / ray)  ──▶  WorkerWrapperBase  ──▶  Worker (gpu_worker.py)
                                                              │ execute_model()
                                                              ▼
                                              GPUModelRunner.execute_model()
                                                ├─ _update_states()         更新持久化批
                                                ├─ _prepare_inputs()        拼 GPU 张量/注意力元数据
                                                ├─ _preprocess()            多模态编码器/嵌入
                                                ├─ model.forward()          真正的网络前向
                                                ├─ compute_logits() + sampler/rejection_sampler
                                                └─ _bookkeeping_sync()      回写采样 token、logprobs
                                                              │
                                              ModelRunnerOutput（采样 token 已在 CPU）
```

---

## 一、Worker：进程内的设备代理

### 1.1 三层类关系

- **`WorkerWrapperBase`**（[worker_base.py:179](../../vllm/v1/worker/worker_base.py#L179)）：代表 executor 中的**一个进程**。它只在该进程里惰性地创建真正的 worker：记住 `rpc_rank/global_rank`，`init_worker()` 时根据 `parallel_config.worker_cls`（字符串 qualname）反射出 worker 类并实例化。`__getattr__` 把所有未识别属性转发给内部 `self.worker`，所以 wrapper 本身看起来就像 worker。
- **`WorkerBase`**（[worker_base.py:38](../../vllm/v1/worker/worker_base.py#L38)）：设备无关的接口，定义了 `init_device / load_model / determine_available_memory / get_kv_cache_spec / initialize_from_config / compile_or_warm_up_model / execute_model / sample_tokens / shutdown` 等抽象方法，以及 LoRA、权重热更新、profiler 等通用契约。
- **`Worker`**（[gpu_worker.py:106](../../vllm/v1/worker/gpu_worker.py#L106)）：GPU 实现，继承 `WorkerBase`，组合一个 `GPUModelRunner`。CPU/XPU 各自有对应的 `cpu_worker.Worker`、`xpu_worker.Worker`。

Wrapper 还承担两件杂事：`update_environment_variables()`（每个 rank 注入不同环境变量）、`_apply_mm_cache()`（用共享内存在进程间传递多模态预处理缓存，见 [worker_base.py:322](../../vllm/v1/worker/worker_base.py#L322)）。此外 `worker_extension_cls` 支持把一个扩展类**动态注入到 worker 的基类**中（[worker_base.py:253-279](../../vllm/v1/worker/worker_base.py#L253-L279)），用于新增 collective_rpc 方法。

### 1.2 完整生命周期

引擎通过 `Executor.collective_rpc(method, ...)` 广播调用。从 [core.py:236-283](../../vllm/v1/engine/core.py#L236-L283) 与 [abstract.py:118-150](../../vllm/v1/executor/abstract.py#L118-L150) 可串出顺序：

| 顺序 | RPC 方法 | 作用 | 实现位置 |
|------|-----------|------|----------|
| 1 | `init_worker`（wrapper） | 反射并构造 Worker，加载插件，设置 vllm_config 上下文 | [worker_base.py:222](../../vllm/v1/worker/worker_base.py#L222) |
| 2 | `update_environment_variables` | 注入 rank 相关环境变量 | [worker_base.py:214](../../vllm/v1/worker/worker_base.py#L214) |
| 3 | `init_device` | 选卡、初始化分布式（NCCL/TP/PP/CP/EC）、设随机种子、拍**初始显存快照**、构造 `GPUModelRunner` | [gpu_worker.py:239](../../vllm/v1/worker/gpu_worker.py#L239) |
| 4 | `load_model` | 用 model_loader 加载权重到 GPU（含 LoRA 包装、drafter、EPLB、CUDAGraphWrapper/UBatchWrapper） | [gpu_worker.py:338](../../vllm/v1/worker/gpu_worker.py#L338) / [gpu_model_runner.py:4852](../../vllm/v1/worker/gpu_model_runner.py#L4852) |
| 5 | `get_kv_cache_spec` | 遍历所有 Attention 层，得到每层的 `KVCacheSpec`（形状/dtype/block_size/MLA…） | [gpu_model_runner.py:7058](../../vllm/v1/worker/gpu_model_runner.py#L7058) |
| 6 | `determine_available_memory` | **profile run**（dummy 前向）测峰值激活，再可选 profile CUDA graph 显存，算出能给 KV cache 的字节数 | [gpu_worker.py:354](../../vllm/v1/worker/gpu_worker.py#L354) |
| 7 | （引擎侧用 spec + 可用显存算出 `KVCacheConfig`，即 num_blocks） | | |
| 8 | `initialize_from_config` | 写回 `num_gpu_blocks`，初始化 KV transfer，按 config **真正分配 KV cache**，初始化 routed-experts capturer、KV 清零元数据 | [gpu_worker.py:539](../../vllm/v1/worker/gpu_worker.py#L539) |
| 9 | `compile_or_warm_up_model` | 对 compile_sizes/cudagraph_sizes 做 dummy run 触发 `torch.compile`/Triton JIT，**捕获 CUDA graph**，最后 warmup sampler/pooler 并激活 Triton JIT 监控 | [gpu_worker.py:574](../../vllm/v1/worker/gpu_worker.py#L574) |
| 10 | （稳态循环）`execute_model`（+ 必要时 `sample_tokens`） | 每 step 执行一次 | [gpu_worker.py:783](../../vllm/v1/worker/gpu_worker.py#L783) |
| — | `shutdown` | 关 KV transfer、profiler、权重传输引擎，释放 model runner 资源 | [gpu_worker.py:1102](../../vllm/v1/worker/gpu_worker.py#L1102) |

#### init_device 的几个要点（[gpu_worker.py:239-334](../../vllm/v1/worker/gpu_worker.py#L239-L334)）

- **DP 调整 local_rank**：非 ray/external_launcher 且单节点时，`local_rank += dp_local_rank * (pp_size * tp_size)`，把数据并行 rank 映射到不同 GPU（[L244-272](../../vllm/v1/worker/gpu_worker.py#L244-L272)）。
- **先初始化分布式再测显存**：NCCL 缓冲要先分配掉，随后 `gc.collect()` + `empty_cache()`，再拍 `MemorySnapshot`，并按 `gpu_memory_utilization` 算出"目标显存" `requested_memory`（[L297-307](../../vllm/v1/worker/gpu_worker.py#L297-L307)）。
- **构造 model runner**：根据 `VLLM_USE_V2_MODEL_RUNNER` 选择 V1/V2。此时只构造对象，还没加载模型。

#### load_model 的显存标签（[gpu_worker.py:338-345](../../vllm/v1/worker/gpu_worker.py#L338-L345)）

加载权重时套了两层上下文：`_maybe_get_memory_pool_context(tag="weights")`（配合 sleep 模式的 CuMemAllocator）和 `_scoped_allocator_max_split(20MiB)`（临时把 CUDA 分配器切片调小以减少碎片）。

### 1.3 determine_available_memory：显存是怎么算出来的

[gpu_worker.py:354-506](../../vllm/v1/worker/gpu_worker.py#L354-L506) 是理解"vLLM 怎么知道能开多少 KV cache"的关键：

1. 如果用户显式设了 `kv_cache_memory_bytes`，仍跑一次 `profile_run()`（为了把 `max_num_batched_tokens` 编译出来），但直接用用户给的字节数；
2. 否则进入 `memory_profiling(init_snapshot, weights_memory=...)` 上下文，执行 `model_runner.profile_run()`，读取 `allocated_bytes.all.peak` 得到 **torch 峰值激活**；
3. CUDA 且开启 cudagraph 时，调 `profile_cudagraph_memory()` 单独估算 graph 占用（它用临时 pool 真捕获几个样例 graph 再清理，见 [gpu_model_runner.py:6049](../../vllm/v1/worker/gpu_model_runner.py#L6049)）；
4. 汇总：
   ```
   available_kv_cache = requested_memory
                        - non_torch_increase      # NCCL/驱动等非 torch 分配
                        - torch_peak_increase     # 峰值激活
                        - weights_memory          # 权重
                        - cudagraph_estimate      # graph 内存（可选）
   ```
5. 打印一段"等效 gpu_memory_utilization"提示（v0.21.0 默认把 cudagraph 内存计入 profile）。

`profile_run()`（[gpu_model_runner.py:5888](../../vllm/v1/worker/gpu_model_runner.py#L5888)）本身做两件事：若有多模态，先用最大规格的 dummy 多模态输入跑一遍视觉编码器并填入 `encoder_cache`；然后 `_dummy_run(max_num_tokens, is_profile=True)` 触发一次完整前向 + 采样/池化。

### 1.4 initialize_from_config：真正分配 KV cache

[gpu_worker.py:539-571](../../vllm/v1/worker/gpu_worker.py#L539-L571)：

- 把 profile 后确定的 `num_blocks` 写回 `cache_config.num_gpu_blocks`；
- `ensure_kv_transfer_initialized()`（KV connector）——必须在分配 KV cache 前；
- sleep 模式下在 `tag="kv_cache"` 的内存池里分配；
- 调 `model_runner.initialize_kv_cache(kv_cache_config)`（详见第三节）；
- 若开启 `enable_return_routed_experts`，初始化 routed-experts capturer；
- 需要时构造 KV 清零元数据（`_init_kv_zero_meta`），且**放在 CuMem 池外**分配，避免 sleep/wake 被回收。

### 1.5 compile_or_warm_up_model：编译与 CUDA graph

[gpu_worker.py:574-727](../../vllm/v1/worker/gpu_worker.py#L574-L727)：

1. 收集 `warmup_sizes`：`VLLM_COMPILE` 模式下，`compile_sizes` 里那些**不在** `cudagraph_capture_sizes` 中的尺寸需要 dummy run 来触发编译；再为每个 `compile_range` 补一个端点；
2. 从大到小对每个 size 调 `_dummy_run(size, skip_eplb=True, remove_lora=False)`——先跑大尺寸，让后续小尺寸复用显存池；
3. `kernel_warmup(self)` 预热融合算子；
4. 非 eager 模式下调 `model_runner.capture_model()` 捕获 CUDA graph，返回实际 graph 显存并与估算值对比；
5. 最后 PP 末级再做一次 `_dummy_run` 来**预热 sampler/pooler 并预分配 logits 缓冲**（故意放在 capture 之后，防止被 `empty_cache` 清掉）；
6. 重置随机种子，激活 Triton JIT 监控（此后意外的 JIT 编译会被当作延迟尖峰报告）。

> V2 runner 的 kernel warmup 走 [gpu/warmup.py](../../vllm/v1/worker/gpu/warmup.py) 的 `warmup_kernels()`：它手工构造两次假的 `SchedulerOutput`（一次 `2+num_spec` token 的 prefill，一次 `1+num_spec` token 的 decode），完整走 `execute_model` + `sample_tokens`，从而 JIT 编译所有 Triton kernel。

### 1.6 execute_model：PP 收发与转发

[gpu_worker.py:783-871](../../vllm/v1/worker/gpu_worker.py#L783-L871) 本身很薄，主要处理**流水并行（PP）**：

- 等上一轮遗留的非阻塞 PP send 完成；
- 若开启 SP（sequence parallel），先调 `_determine_batch_execution_and_padding` 算 batch 描述，决定 all-gather 哪些张量；
- 非首 rank：`irecv_tensor_dict` 从上一 stage 收中间张量，包成 `AsyncIntermediateTensors`（懒同步，访问 `.tensors` 时才 wait handle，见 [L74-103](../../vllm/v1/worker/gpu_worker.py#L74-L103)）；
- 调 `model_runner.execute_model(scheduler_output, intermediate_tensors)`；
- 末 rank 返回 `ModelRunnerOutput`；非末 rank 拿到 `IntermediateTensors` 后用 `isend_tensor_dict` **非阻塞**发给下一 stage，返回 `None`。

注意一个两段式设计：`model_runner.execute_model()` 在结构化输出（grammar bitmask）需要时会**返回 None**，并把中间状态存到 `self.execute_model_state`；引擎紧接着调用 `sample_tokens(grammar_output)` 才真正采样并产出输出（[gpu_model_runner.py:3855-4203](../../vllm/v1/worker/gpu_model_runner.py#L3855-L4203) 与 [4206](../../vllm/v1/worker/gpu_model_runner.py#L4206)）。这是因为 grammar 掩码要在引擎/调度器侧根据结构化输出规范生成，GPU 前向得先跑完、采样延后。

Worker 还提供 `sleep/wake_up`（[gpu_worker.py:160-199](../../vllm/v1/worker/gpu_worker.py#L160-L199)）通过 CuMemAllocator 释放/重建权重与 KV cache 显存，以及一整套在线权重热更新（`init_weight_transfer_engine / start_weight_update / update_weights / finish_weight_update`，用于训练-推理联动）。

---

## 二、GPUModelRunner：单卡执行的大脑

`GPUModelRunner`（[gpu_model_runner.py:399](../../vllm/v1/worker/gpu_model_runner.py#L399)）继承三个 mixin：`LoRAModelRunnerMixin`、`KVConnectorModelRunnerMixin`、`ECConnectorModelRunnerMixin`。它持有一长串状态，可归为几类。

### 2.1 __init__ 里分配了什么

[gpu_model_runner.py:402-868](../../vllm/v1/worker/gpu_model_runner.py#L402-L868)：

**配置与派生量**：各 `*_config`、`device`、`dtype`、`kv_cache_dtype`、`num_query_heads`、`is_pooling_model`、`max_num_tokens`（=`max_num_batched_tokens`）、`max_num_reqs`、`max_model_len`、cascade attention 开关、M-RoPE/XD-RoPE 开关、DCP rank 等。

**子组件**：
- `self.sampler = Sampler(...)`（[L488](../../vllm/v1/worker/gpu_model_runner.py#L488)）；
- 投机解码 `self.drafter`（ngram / draft model / eagle / eagle3 / medusa / dflash / gemma4 / suffix / extract_hidden_states 之一）+ `self.rejection_sampler`（[L522-590](../../vllm/v1/worker/gpu_model_runner.py#L522-L590)）；
- `self.input_batch = InputBatch(...)`（持久化批，见第四节）；
- `self.cudagraph_dispatcher = CudagraphDispatcher(...)`（按 batch 描述分发到对应 graph）；
- 多模态 `self.mm_budget`、`self.encoder_cache`（mm_hash→编码器输出）、`late_interaction_runner`。

**请求状态**：`self.requests: dict[str, CachedRequestState]`、`self.num_prompt_logprobs`。

**持久化 GPU 缓冲**（按 `max_num_tokens` / `max_num_reqs` 预分配，CUDA graph 依赖它们地址稳定）：
```python
self.input_ids          # [max_num_tokens] int32   本 step 的输入 token
self.positions          # [max_num_tokens] int64   位置 id
self.query_start_loc    # [max_num_reqs+1] int32   每个请求在拼接序列中的起止（cumsum）
self.seq_lens           # [max_num_reqs] int32     本 step 后的序列长度（乐观值）
self.num_computed_tokens# [max_num_reqs] int32     已算过的 KV 长度
self.req_indices        # [max_num_tokens] int64   每个 token 属于哪个 req
self.num_scheduled_tokens   # [max_num_reqs] int32
self.inputs_embeds      # [max_num_tokens, hidden_size]  多模态/prompt_embeds 用
self.is_token_ids       # [max_num_tokens] bool
self.mrope_positions    # (3, max_num_tokens+1) int64   Qwen2-VL 等
```
这些大多是 `CpuGpuBuffer`（[v1/utils.py:108](../../vllm/v1/utils.py#L108)）——同时持有 CPU ndarray 和 GPU tensor，`copy_to_gpu(n)` 做一次（可分页/非阻塞）传输。

**异步/投机相关的流与事件**：`async_output_copy_stream`、`prepare_inputs_event`、`transfer_event`、`draft_token_ids_*`、`valid_sampled_token_count_*`、`num_accepted_tokens` 等，用于 async scheduling 下把 D2H 拷贝和下一 step 的 CPU 准备重叠。

**KV cache 占位**：`self.kv_caches: list[torch.Tensor]`、`self.attn_groups: list[list[AttentionGroup]]`、`self.shared_kv_cache_layers`（跨层 KV 共享）。

### 2.2 load_model：加载并包装模型

[gpu_model_runner.py:4852-5023](../../vllm/v1/worker/gpu_model_runner.py#L4852-L5023)：

1. 用 `DeviceMemoryProfiler` 包住，记录权重显存 `model_memory_usage`；
2. `get_model_loader(load_config).load_model(...)` 得到 `self.model`（支持 dummy 权重、各种 load_format）；
3. 若启用 LoRA，`self.model = self.load_lora_model(...)`；
4. 加载 drafter 模型，MoE drafter 也纳入 EPLB；
5. EAGLE-3 设置辅助隐状态层；
6. MoE + EPLB：`eplb_state.add_model(...)`，异步 EPLB 启动后台循环；
7. `STOCK_TORCH_COMPILE` 模式直接 `model.compile(fullgraph=True, backend=...)` 后返回；
8. 否则按 `cudagraph_mode` 用 `CUDAGraphWrapper`（FULL graph）或 `UBatchWrapper`（微批）包装模型——这层包装负责运行时判断当前 batch 是否命中已捕获的 graph，否则回退 eager。
9. `get_offloader().post_init()`。

### 2.3 执行主链路：execute_model

[gpu_model_runner.py:3855-4203](../../vllm/v1/worker/gpu_model_runner.py#L3855-L4203) 是稳态每步的入口。整理后的主干：

```python
def execute_model(scheduler_output, intermediate_tensors=None):
    # 0. 异常状态检查、ngram_gpu 下 copy scheduler_output、KV connector 处理抢占
    ...
    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

    with record_function("preprocess"), self.synchronize_input_prep():
        # 1. 更新持久化批：加/删/恢复请求，刷新 block_table、采样元数据
        deferred_state_corrections_fn = self._update_states(scheduler_output)

        # 2. （PD 分离的 producer）只跑编码器就返回空输出
        if has_ec_transfer() and not is_consumer:
            self._execute_mm_encoder(...); return make_empty_encoder_model_runner_output(...)

        # 3. 空批：可能 dummy_run 协调 DP，或走 KV connector no-forward 路径
        if not num_scheduled_tokens: return EMPTY_MODEL_RUNNER_OUTPUT

        # 4. 把本 step 拼成模型输入张量 + 注意力元数据
        logits_indices, spec_decode_metadata = self._prepare_inputs(scheduler_output, num_scheduled_tokens_np)

        # 5. cascade attention 前缀长度
        cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(...)

        # 6. 决定执行方式：用哪个 cudagraph_mode、padding 到多少、是否微批、DP token 数
        cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, cudagraph_stats = \
            self._determine_batch_execution_and_padding(...)

        # 7. Mamba SSM 状态预处理（拷贝 conv/state 到新 block）
        if cache_config.mamba_cache_mode == "align": mamba_utils.preprocess_mamba(...)

        # 8. 计算每个 KV group 的 slot mapping，构造所有层的 AttentionMetadata
        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(...)
        attn_metadata, spec_common_meta = self._build_attention_metadata(...)

        # 9. 多模态编码器 + 输入嵌入，得到 input_ids / inputs_embeds / positions / model_kwargs
        input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_out = \
            self._preprocess(scheduler_output, num_tokens_padded, intermediate_tensors)

    # 10. 真正前向（set_forward_context 把 attn_metadata/slot_mapping 放进全局上下文，
    #     供 torch.ops.vllm.* 与各 attention 层读取；KV connector output 作为上下文）
    with set_forward_context(attn_metadata, ..., slot_mapping=slot_mappings), \
         self.maybe_get_kv_connector_output(...):
        model_output = self._model_forward(input_ids=..., positions=...,
                                           intermediate_tensors=..., inputs_embeds=..., **model_kwargs)

    with record_function("postprocess"):
        hidden_states = model_output            # EAGLE-3 时是 (hidden, aux_hidden)
        # 非末 PP rank：返回 IntermediateTensors 给下一 stage
        if not get_pp_group().is_last_rank: return hidden_states
        if self.is_pooling_model: return self._pool(...)
        # 末级：取需要采样位置的 hidden states，算 logits
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

    # 11. 保存两段式状态，execute_model 返回 None，由 sample_tokens() 继续
    self.execute_model_state = ExecuteModelState(scheduler_output, logits, spec_decode_metadata, ...)
    if deferred_state_corrections_fn: deferred_state_corrections_fn()
    return None
```

要点：

- **`_update_states()` 返回一个延迟回调**：async spec decode 下，先乐观假设所有 draft 被接受，等模型前向发出后再用真实 `valid_sampled_token_count` 校正 `num_computed_tokens`，不阻塞异步流水线（[gpu_model_runner.py:1415-1445](../../vllm/v1/worker/gpu_model_runner.py#L1415-L1445)）。
- **`logits_indices`**：指示拼接的 token 序列里哪些位置需要算 logits。普通情况就是每个请求最后一个 token（`query_start_loc[1:]-1`）；投机解码时由 `SpecDecodeMetadata` 给出每个 draft+target 的位置（[L2071-2109](../../vllm/v1/worker/gpu_model_runner.py#L2071-L2109)）。
- **`_determine_batch_execution_and_padding`**（[L3621](../../vllm/v1/worker/gpu_model_runner.py#L3621)）是执行路径的路由器：判断是否 uniform decode、padding 到哪个 cudagraph bucket、是否需要 cascade/ubatch，产出 `BatchDescriptor`。
- **`set_forward_context`**（[L4100](../../vllm/v1/worker/gpu_model_runner.py#L4100)）是模型层与 runner 解耦的关键：Attention 层不直接收 metadata 参数，而是从这个全局上下文里取，从而兼容 `torch.compile`。

### 2.4 _update_states：把调度结果灌进持久化批

[gpu_model_runner.py:1073-1447](../../vllm/v1/worker/gpu_model_runner.py#L1073-L1447) 做增量状态同步：

1. **移除**：`finished_req_ids` 从 `self.requests` 和 `input_batch` 删除；释放 routed-experts 缓冲、encoder cache；
2. **清零新分配的物理块**（`_zero_block_ids`，避免 NaN 污染注意力/SSM）；
3. **移除本 step 未调度的请求**（preempt 或未排到的），但保留 `CachedRequestState` 以便之后恢复；
4. **加入新请求**：为每个 `scheduled_new_reqs` 构造 `CachedRequestState`（prompt、block_ids、sampling/pooling params、generator、lora），注册 prompt_logprobs、M-RoPE 位置等，收集到 `reqs_to_add`；
5. **更新运行中/恢复的请求**：写 `num_computed_tokens`、追加/替换 `block_ids`、更新 `token_ids_cpu`（非末 PP rank 时 scheduler 会回传采样 token；末 rank 直接用本地缓存）、写入 spec token；
6. **批量 add + condense + reorder + refresh_metadata**：新请求填入空槽位，`condense()` 紧凑化（见第四节），`_may_reorder_batch()` 允许注意力后端重排，`refresh_metadata()` 重建采样元数据；
7. ngram_gpu 增量更新 GPU 张量。

### 2.5 _prepare_inputs：拼出 GPU 张量

[gpu_model_runner.py:1815-2124](../../vllm/v1/worker/gpu_model_runner.py#L1815-L2124) 是"调度→张量"的核心翻译器。主要步骤：

1. **先异步提交 block table 的 H2D 拷贝**（`commit_block_table`），与后面 CPU 计算重叠；
2. `req_indices = np.repeat(arange(num_reqs), num_scheduled_tokens)`，`cu_num_tokens = cumsum(...)`，并在 `query_pos` 里写出每请求内的局部位置 `[0,1,...,len-1]`；
3. **positions** = `num_computed_tokens[req_idx] + query_pos`；M-RoPE/XD-RoPE 走专门计算；
4. **取 input_ids**：用 `token_indices = positions + req_idx * max_model_len` 把 2D 的 `token_ids_cpu[req, pos]` 打平，`torch.index_select` 一次性 gather 到 `input_ids.cpu`（比 `np.take` 快，[L1872-1880](../../vllm/v1/worker/gpu_model_runner.py#L1872-L1880)）；prompt embeddings 单独填到 `inputs_embeds`；
5. 写 `query_start_loc`、**乐观 seq_lens**（`num_computed + num_scheduled`，假设 draft 全中）、`discard_request_mask`（chunked prefill 等不该采样的请求）；
6. 处理 async spec decode 的 `num_computed_tokens` GPU 校正；
7. 把 `req_indices/query_pos/num_scheduled_tokens` 拷到 GPU，在 GPU 上算 `positions`、`seq_lens`；
8. **`block_table.compute_slot_mapping(...)`**：用 Triton kernel 把 `(block_table, positions)` 转成每个 token 对应的 `slot_mapping`（KV cache 物理槽位），这是写 KV 的地址；
9. `_prepare_input_ids()`：async scheduling 下把上一 step 缓存在 GPU 的采样 token 直接 scatter 进 `input_ids`，避免绕回 CPU；
10. 投机解码：构造 `num_draft_tokens/num_decode_draft_tokens`，调 `_calc_spec_decode_metadata` 得到 `logits_indices`；否则 `logits_indices = query_start_loc[1:]-1`；
11. LoRA：`set_active_loras(...)` 热切换。

> 关键设计：**token_ids 存在 CPU 的 `[max_num_reqs, max_model_len]` 大矩阵里**，每步只 gather 需要的那一小段到 GPU。这样请求状态更新（追加新采样 token）全在 CPU 端完成，GPU 端只看到扁平的拼接序列。

### 2.6 _build_attention_metadata

[gpu_model_runner.py:2126](../../vllm/v1/worker/gpu_model_runner.py#L2126) 遍历 `self.attn_groups`，对每个 KV cache group / attention group 取对应的 `AttentionMetadataBuilder`，灌入公共的 `CommonAttentionMetadata`（seq_lens、query_start_loc、block_table、slot_mapping、cascade prefix、DCP/PCP 信息等），产出每层专属的 metadata。它还处理 ubatch 切片（`split_attn_metadata`）、prefill/decode 重排（`reorder_batch_to_split_decodes_and_prefills`）、spec decode 的通用 metadata。产物最终通过 `set_forward_context` 暴露给各 Attention 层（呼应第 03 篇）。

### 2.7 _preprocess / _model_forward / _sample / _bookkeeping_sync

- **`_preprocess`**（[L3279](../../vllm/v1/worker/gpu_model_runner.py#L3279)）：首 rank 且多模态时跑 `_execute_mm_encoder` + `_gather_mm_embeddings`，统一用 `model.embed_input_ids(...)` 把 token id 和软嵌入（视觉 embedding）融合成 `inputs_embeds`；纯文本模型则直接传 `input_ids`（把 embedding 层留在 CUDA graph 内以获得更好性能，见 [L3353-3360](../../vllm/v1/worker/gpu_model_runner.py#L3353-L3360)）。非首 rank 收齐 `intermediate_tensors`。encoder-decoder 模型在此跑 encoder。
- **`_model_forward`**（[L3568](../../vllm/v1/worker/gpu_model_runner.py#L3568)）：薄薄一层 `self.model(...)`，单独留出以便子类/编译只检查这一个方法。
- **`_sample`**（[L3397](../../vllm/v1/worker/gpu_model_runner.py#L3397)）：无投机时 `self.sampler(logits, sampling_metadata)`；有投机时 `self.rejection_sampler(spec_metadata, None, logits, sampling_metadata)`。
- **`_bookkeeping_sync`**（[L3427](../../vllm/v1/worker/gpu_model_runner.py#L3427)）：处理 `discard_request_mask`、把采样 token 写回 `token_ids_cpu` 和 `requests[...].output_token_ids`、async 模式下把采样 token 缓存在 GPU（`prev_sampled_token_ids`）供下一 step scatter、计算 prompt logprobs，返回组装 `ModelRunnerOutput` 所需的各部分。

`sample_tokens(grammar_output)`（[L4206](../../vllm/v1/worker/gpu_model_runner.py#L4206)）在 `execute_model` 返回 None 后被调用：取出 `execute_model_state`，应用 grammar bitmask，跑 `_sample`，再跑 `_bookkeeping_sync` 组装出 `ModelRunnerOutput`（非异步）或 `AsyncGPUModelRunnerOutput`（异步，在独立 stream 上非阻塞 D2H，[L232-297](../../vllm/v1/worker/gpu_model_runner.py#L232-L297)）。

### 2.8 KV cache 初始化链路

引擎把 `KVCacheConfig`（含每个 group 的 num_blocks）传下来后：

```
Worker.initialize_from_config
  └─ GPUModelRunner.initialize_kv_cache(kv_cache_config)      # L6922
       ├─ may_add_encoder_only_layers_to_kv_cache_config()    # encoder-only 层单独成组
       ├─ maybe_add_kv_sharing_layers_to_kv_cache_groups()    # 跨层共享 KV
       ├─ initialize_attn_backend(kv_cache_config)            # L6335：按层选 backend，构造 attn_groups
       │     └─ 每组按 (backend_cls, kv_cache_spec) 分桶 → AttentionGroup
       ├─ initialize_mamba_ssu_backend(...)
       ├─ prepare_kernel_block_sizes(...)                     # 协商 kernel block size（可能切分）
       ├─ initialize_metadata_builders(...)                   # 每个 attn_group 建 builder
       ├─ may_reinitialize_input_batch(...)                   # block_size 变了就重建 InputBatch
       └─ initialize_kv_cache_tensors(...)                    # L6839：真正分配显存
             ├─ use_uniform_kv_cache? allocate_uniform_kv_caches  # 跨层统一大 buffer
             │   （KV connector、所有层 dtype/shape 一致时）
             └─ _allocate_kv_cache_tensors + _reshape_kv_cache_tensors  # 通用：按 backend.get_kv_cache_shape
             └─ bind_kv_cache(kv_caches, static_forward_context, self.kv_caches)
```

`bind_kv_cache`（[utils.py:457](../../vllm/v1/worker/utils.py#L457)）做两件事：把张量按层序填入 `runner.kv_caches` 列表，并把每个 Attention 层对象的 `.kv_cache` 属性指向对应张量——模型前向时 Attention 层就通过这个引用访问 KV cache。

**`AttentionGroup`**（[utils.py:221](../../vllm/v1/worker/utils.py#L221)）是一个数据类，记录同一 `(backend, kv_cache_spec)` 下的若干层名，以及为它们创建的 metadata builder 列表（ubatching 时每个微批一个 builder，避免持久缓冲冲突）。

**block size 协商**：调度器的分配 block size 可能比 attention kernel 支持的大。`select_common_block_size`/`prepare_kernel_block_sizes`（[utils.py:260-340](../../vllm/v1/worker/utils.py#L260-L340)）找一个所有 backend 都支持且整除 manager block size 的 kernel block size；`BlockTable` 用 `blocks_per_kv_block` 做虚拟拆分（例如 32 token 的分配块映射到 2 个 16 token kernel 块，[block_table.py:53-66](../../vllm/v1/worker/block_table.py#L53-L66)）。

### 2.9 CUDA graph 捕获

- **`capture_model()`**（[L6150](../../vllm/v1/worker/gpu_model_runner.py#L6150)）：从 `cudagraph_dispatcher.get_capture_descs()` 拿到要捕获的 `(runtime_mode, [BatchDescriptor])`，大尺寸优先；在 `graph_capture()` 上下文 + 冻结 GC 中，对每个 desc 调 `_capture_cudagraphs → _warmup_and_capture`：先 warmup 几次（`cudagraph_num_of_warmups`），再以 `is_graph_capturing=True` 跑一次 `_dummy_run` 真正录制；可选捕获视觉编码器 graph；最后 `lock_workspace()` 锁死工作区大小。
- **`profile_cudagraph_memory()`**（[L6049](../../vllm/v1/worker/gpu_model_runner.py#L6049)）：正式分配前用临时 pool 试捕前两个 graph，按"首次捕获内存 + (N-1)×每图增量"估算总占用，FULL/PIECEWISE 共享 pool 取 max 避免重复计算。
- **`_dummy_run(num_tokens, ...)`**（[L5330](../../vllm/v1/worker/gpu_model_runner.py#L5330)）：构造假的输入张量（含假多模态、假 spec token、假 LoRA），完整走一遍 `_prepare_inputs → _preprocess → forward → 采样/pooler`，是编译/warmup/捕获共用的驱动函数。

### 2.10 其他能力（速览）

| 能力 | 入口 | 说明 |
|------|------|------|
| 多模态编码器 | `_execute_mm_encoder`/`_gather_mm_embeddings`（[L2775](../../vllm/v1/worker/gpu_model_runner.py#L2775)）、`gpu/mm/encoder_runner.py` | 按 mm_hash 缓存编码器输出，支持 encoder budget、pruning、顺序视频编码 |
| 池化模型 | `_pool`（[L3195](../../vllm/v1/worker/gpu_model_runner.py#L3195)）、`gpu/pool/pooling_runner.py`、`late_interaction_runner.py` | 对 finished 请求取隐状态做 pool；late-interaction（ColBERT 类）单独处理 |
| 投机解码 | `propose_draft_token_ids`（[L4585](../../vllm/v1/worker/gpu_model_runner.py#L4585)）、`gpu/spec_decode/` | ngram/eagle/medusa/draft model 提案 + rejection sampler 校验 |
| 结构化输出 | `gpu/structured_outputs.py`、`apply_grammar_bitmask` | 在 logits 上应用 grammar 允许 token 位掩码 |
| LoRA | `lora_model_runner_mixin.py` | 热加载/移除/pin LoRA，按批构建 `LoRAMapping` |
| KV connector | `kv_connector_model_runner_mixin.py` | PD 分离的 KV 传输、`kv_connector_no_forward`、跨层统一 KV cache 分配 |
| EC connector | `ec_connector_model_runner_mixin.py` | 预填充/解码分离场景下的编码器输出传输 |
| EPLB | `eplb_step`/`setup_eplb_from_mapping`（[L3162](../../vllm/v1/worker/gpu_model_runner.py#L3162)） | MoE 专家并行负载均衡，后台调整专家放置 |
| Sleep/Wake | Worker 层 `sleep/wake_up` + `post_kv_cache_wake_up` | CuMemAllocator 释放/重建显存 |
| 在线权重更新 | Worker 的 `update_weights` 系列 | 从训练端 NCCL 广播权重，支持 checkpoint/kernel 两种格式 |

---

## 三、InputBatch：持久化批

`InputBatch`（[gpu_input_batch.py:91](../../vllm/v1/worker/gpu_input_batch.py#L91)）是 Worker 侧最核心的数据结构。它维护一个**固定容量（`max_num_reqs`）的类 SOA 批**，每个请求占一个槽位（req_index），连续 step 之间增量更新，而不是每步重建。

### 3.1 CachedRequestState

[gpu_input_batch.py:34-88](../../vllm/v1/worker/gpu_input_batch.py#L34-L88)：每个请求的"真相源"（CPU 端 dataclass）：

```python
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params / pooling_params
    generator: torch.Generator | None       # 带 seed 的随机源
    block_ids: tuple[list[int], ...]        # 每个 KV group 的物理块
    num_computed_tokens: int                # 已算 KV 长度
    output_token_ids: list[int]             # 已生成 token
    mrope_positions / xdrope_positions      # 多模态旋转位置
    lora_request / prompt_embeds / prompt_is_token_ids
    in_progress_prompt_logprobs_cpu         # 跨 chunk 累积的 prompt logprobs
    prev_num_draft_len                      # async spec decode 用
    pooling_states
```

`num_tokens = num_prompt_tokens + len(output_token_ids)`；`get_token_id(idx)` 按位置取 prompt 或生成 token。

### 3.2 InputBatch 持有什么

构造函数（[L92-301](../../vllm/v1/worker/gpu_input_batch.py#L92-L301)）预分配：

- **索引**：`_req_ids: list[str|None]`（长度=当前批容量，空槽为 None）、`req_id_to_index: dict`；
- **token 存储**：`token_ids_cpu_tensor = zeros(max_num_reqs, max_model_len, int32)` + numpy 视图 `token_ids_cpu`；以及 `is_token_ids`（区分 token id 与 prompt embedding）；`num_tokens_no_spec`、`num_prompt_tokens`、`num_computed_tokens_cpu`（每个都是 CPU tensor + numpy 视图，多数 pin_memory）；
- **block table**：`MultiGroupBlockTable`（见第四节）；
- **采样参数**（CPU ndarray + GPU tensor 成对）：`temperature/top_p/top_k/frequency_penalties/presence_penalties/repetition_penalties`，以及 `greedy_reqs/random_reqs/top_p_reqs/...` 等集合；
- **LoRA**：`request_lora_mapping`、`lora_id_to_request_ids`；
- **生成器**：`generators: dict[req_index, torch.Generator]`；
- **logprobs / allowed tokens / bad words / logits processors** 状态；
- **spec decode**：`num_accepted_tokens_cpu`、`spec_token_ids`；
- **pooling**：`pooling_params/pooling_states`；
- **async scheduling**：`prev_sampled_token_ids`（上一 step GPU 采样结果）、`prev_req_id_to_index`。

### 3.3 关键方法

| 方法 | 作用 |
|------|------|
| `add_request(req_state)`（[L335](../../vllm/v1/worker/gpu_input_batch.py#L335)） | 找空槽（优先小索引），把 prompt token 写入 `token_ids_cpu`，初始化温度/top_p/penalties、generator、lora mapping、spec token |
| `remove_request(req_id)`（[L510](../../vllm/v1/worker/gpu_input_batch.py#L510)） | 清槽位、block table 行、采样状态；删 `req_id_to_index` |
| `swap_states(i1,i2)`（[L566](../../vllm/v1/worker/gpu_input_batch.py#L566)） | 交换两槽的所有张量切片/字典条目（reorder/condense 用） |
| `condense()`（[L683](../../vllm/v1/worker/gpu_input_batch.py#L683)） | 移除请求后可能留下空槽缺口，把后面的有效请求前移，使活动槽位连续紧凑（attention kernel 友好） |
| `refresh_metadata()`（[L811](../../vllm/v1/worker/gpu_input_batch.py#L811)） | 调各 logits processor 的 staged writes，重建 `SamplingMetadata` |
| `_make_sampling_metadata()`（[L831](../../vllm/v1/worker/gpu_input_batch.py#L831)） | 把所有 GPU 采样张量 + 调度信息打包成 `SamplingMetadata` 给 sampler |
| `update_req_spec_token_ids`（[L483](../../vllm/v1/worker/gpu_input_batch.py#L483)） | 把本 step 的 draft token 写入 `token_ids_cpu`/`spec_token_ids` |
| `update_async_output_token_ids` / `update_async_spec_token_ids` | async 模式下，等 D2H 拷贝真正完成后再把 token id 回填到 CPU（penalty/bad_words 需要） |
| `make_lora_inputs`（[L976](../../vllm/v1/worker/gpu_input_batch.py#L976)） | 构建 `LoRAMapping`（每 token 对应哪个 lora，prompt/decode 索引映射） |
| `all_greedy/all_random/no_top_p/no_penalties/...` | 批量属性，让 sampler 能跳过不需要的 kernel（快速路径） |

### 3.4 V2 的重构

V2 把这块拆成了 [gpu/input_batch.py](../../vllm/v1/worker/gpu/input_batch.py) 的 `InputBuffers` + `InputBatch`，[gpu/states.py](../../vllm/v1/worker/gpu/states.py) 的 `RequestState`，以及一堆 Triton kernel（`prepare_prefill_inputs / prepare_pos_seq_lens / combine_sampled_and_draft_tokens / post_update` 等）把原本 CPU 端的批更新逻辑搬到 GPU 上做，并通过 `staged writes` 模式累积本 step 变更再批量提交。`gpu/model_states/` 还抽象出 `ModelState` 接口，让 mamba-hybrid、whisper 等特殊模型插入自定义的输入准备/注意力元数据。

---

## 四、BlockTable：逻辑块到物理块

分页 KV cache 的核心映射由 `BlockTable` 维护。

### 4.1 单组 BlockTable

[block_table.py:18](../../vllm/v1/worker/block_table.py#L18)：

- 持有一个 `CpuGpuBuffer`，形状 `[max_num_reqs, max_num_blocks_per_req]`（int32），即每个请求一行，行内是该请求占用的**物理块 id 序列**；另有 `num_blocks_per_row` 记录每行已用块数。
- `append_row(new_block_ids, row_idx)`（[L102](../../vllm/v1/worker/block_table.py#L102)）：调度器新分配块时追加。若 `use_hybrid_blocks`（manager block ≠ kernel block），先用 `map_to_kernel_blocks` 把一个分配块拆成多个 kernel 块 id（[L174-201](../../vllm/v1/worker/block_table.py#L174-L201)）。
- `commit_block_table(num_reqs)`（[L166](../../vllm/v1/worker/block_table.py#L166)）：把整张表 H2D 拷贝。
- **`compute_slot_mapping(num_reqs, query_start_loc, positions)`**（[L141](../../vllm/v1/worker/block_table.py#L141)）：核心 Triton kernel `_compute_slot_mapping_kernel`，对每个 token 计算
  ```
  block_idx = positions[token] // block_size
  offset     = positions[token] %  block_size
  slot       = block_table[req, block_idx] * block_size + offset
  ```
  并考虑 PCP/DCP（上下文并行）rank/interleave，把不归属本 rank 的槽位填成 `PAD_SLOT_ID`。`slot_mapping` 就是写 KV cache 的物理地址向量。
- `move_row/swap_row/clear_row`：condense/reorder/抢占时用。

### 4.2 MultiGroupBlockTable

[block_table.py:223](../../vllm/v1/worker/block_table.py#L223)：不同 KV cache group（如注意力层 + Mamba SSM、或不同 block size）各有一张 `BlockTable`，对外提供同名方法批量转发。`InputBatch.block_table` 就是它。调度器传来的 `block_ids`/`new_block_ids` 是 `tuple[list[int], ...]`，每个 group 一份。

> V2 的 [gpu/block_table.py](../../vllm/v1/worker/gpu/block_table.py) 用 `BlockTables` + 指针表（`_make_ptr_tensor`）把多张表的设备指针打包给 kernel，并通过 `append_block_ids`（staged）+ `apply_staged_writes` + `gather_block_tables` 把行收集成连续张量，配合 `gpu/input_batch.py` 的 GPU 端更新路径。

---

## 五、采样、池化、多模态与投机

### 5.1 采样

V1 默认用 [v1/sample/sampler.py](../../vllm/v1/sample/sampler.py) 的 `Sampler(nn.Module)`，在 `GPUModelRunner._sample` 中调用：输入 `logits + SamplingMetadata`，依次做 logits processors（grammar/allowed tokens/bad words/logit bias）、penalties（repetition/frequency/presence，需 token 历史）、temperature、top-k/top-p、min-p、gumbel 采样/greedy，并按需算 logprobs。

V2 把采样器拆成 [gpu/sample/](../../vllm/v1/worker/gpu/sample/) 下的多个状态组件：`SamplingStates`、`PenaltiesState`、`LogitBiasState`、`BadWordsState`、`LogprobTokenIdsState`，由 [gpu/sample/sampler.py](../../vllm/v1/worker/gpu/sample/sampler.py) 的 `Sampler.__call__` 组合，所有状态都跟着 `RequestState` 走 staged-write 更新。

**投机解码**：`RejectionSampler`（[v1/sample/rejection_sampler.py](../../vllm/v1/sample/rejection_sampler.py)）对 draft token 做采样概率与目标概率的拒绝采样校验，输出 `[num_reqs, 1+num_spec]` 的张量；`parse_output` 把它还原成每请求变长的 accepted token 列表。`gpu/spec_decode/` 下有 EAGLE（1/3）、ngram GPU 张量更新、rejection sampler 工具等。

### 5.2 池化

`GPUModelRunner._pool`（[L3195](../../vllm/v1/worker/gpu_model_runner.py#L3195)）处理 embedding/reward/classification 等非生成模型：对 finished 请求取对应隐状态，调 `model.pooler(...)`，再用 `_copy_pooler_output_to_cpu` 非阻塞拷回 CPU。Late-interaction 模型（ColBERT 类，输出变长向量序列）由 `gpu/pool/late_interaction_runner.py` 管理，在请求结束时收集所有 token 的表示。

### 5.3 多模态编码器

- `gpu/mm/encoder_runner.py` 的 `EncoderRunner` 负责调度视觉/音频编码器的前向（支持批量合并、budget 控制、顺序视频编码、编码器 CUDA graph）；
- `gpu/mm/encoder_cache.py` 维护 mm_hash→编码器输出缓存（与调度器侧的 `EncoderCacheManager` 配合，避免重复编码相同图片）；
- `gpu/mm/rope.py` 处理多模态 RoPE；
- `encoder_cudagraph.py`（worker 根目录）为支持 `SupportsEncoderCudaGraph` 的视觉模型单独捕获编码器 graph（[capture_model 中 L6164-6185](../../vllm/v1/worker/gpu_model_runner.py#L6164-L6185)）；
- 编码后的 embedding 通过 `_gather_mm_embeddings` 按 `PlaceholderRange` 散布到对应 token 位置，再经 `model.embed_input_ids` 与文本 embedding 融合。

---

## 六、工作区、微批与异步

### 6.1 WorkspaceManager

[workspace.py:31](../../vllm/v1/worker/workspace.py#L31) 是一个全局单例，管理一块可**按需扩容、CUDA graph 捕获后锁定**的临时 GPU 缓冲区，供 flash-attn/cutlass 等 kernel 存放中间结果。`init_workspace_manager(device, num_ubatches)` 在 `init_device` 时创建（DBO 开 2 份），`get_workspace(rank, required_bytes)` 取张量并自动扩容，`lock_workspace()` 在 graph 捕获后禁止再扩容（否则地址变化会让 graph 失效）。

### 6.2 微批 / DBO（Disaggregated Batch Overlap）

- [ubatching.py](../../vllm/v1/worker/ubatching.py) 提供 `UBatchContext`：在同一个 stream 上把一个 batch 切成多个 ubatches，用 `switch_to_compute/switch_to_comm` 事件机制让**计算与通信（TP/PP 集合通信）重叠**；`dbo_register_recv_hook` 注册接收回调。
- [gpu_ubatch_wrapper.py](../../vllm/v1/worker/gpu_ubatch_wrapper.py) 的 `UBatchWrapper` 是模型包装器（类似 CUDAGraphWrapper），内部按 ubatch 切分输入并驱动 `UBatchContext`。
- [ubatch_utils.py](../../vllm/v1/worker/ubatch_utils.py) 的 `UBatchSlices`、`maybe_create_ubatch_slices`、`split_attn_metadata` 负责把 token 与注意力元数据切成微批。

### 6.3 异步调度与输出

- `scheduler_config.async_scheduling=True` 时，runner 在独立 CUDA stream 上做采样结果的 D2H：`AsyncGPUModelRunnerOutput`（[L232](../../vllm/v1/worker/gpu_model_runner.py#L232)）在构造时立即发起 `non_blocking` 拷贝并记录 event，主循环可以立刻开始下一 step；引擎真正取结果时调 `get_output()` 等 event 完成。
- 上一 step 采样的 token 不必回 CPU，直接缓存在 `input_batch.prev_sampled_token_ids`，下一 step 在 `_prepare_input_ids` 里用 scatter 写回 `input_ids`（[L1641](../../vllm/v1/worker/gpu_model_runner.py#L1641)），并通过 `prev_positions` 处理 batch 重排。
- `synchronize_input_prep()`（[L3553](../../vllm/v1/worker/gpu_model_runner.py#L3553)）用 event 保证复用的 CPU 张量在上一 step 的异步传输结束后才被覆盖。
- [gpu/async_utils.py](../../vllm/v1/worker/gpu/async_utils.py) 是 V2 的对应实现。

### 6.4 KV 清零与显存安全

`utils.py` 的 `KVBlockZeroer`（[L80](../../vllm/v1/worker/utils.py#L80)）维护一个清零 kernel 流和块 id 缓冲，调度器每步把新分配的物理块通过 `new_block_ids_to_zero` 传下来，runner 在 `_update_states` 里调 `_zero_block_ids`（[L1051](../../vllm/v1/worker/gpu_model_runner.py#L1051)）异步清零，避免新块里的 NaN/脏数据污染注意力或 SSM。这是分页 KV cache 正确性的重要保障。

---

## 七、CPU / XPU 后端（对照）

- [cpu_worker.py](../../vllm/v1/worker/cpu_worker.py) + [cpu_model_runner.py](../../vllm/v1/worker/cpu_model_runner.py)：CPU 推理路径，结构与 GPU 类似但没有 CUDA graph，注意力走 CPU 后端；支持权重的 zero-copy 映射。
- [xpu_worker.py](../../vllm/v1/worker/xpu_worker.py) + [xpu_model_runner.py](../../vllm/v1/worker/xpu_model_runner.py)：Intel XPU，复用大量 GPU runner 的逻辑，主要在设备/内存/分布式 API 上做适配。

设备无关的契约全部定义在 `WorkerBase`，新增硬件后端只需实现这组接口并提供对应的 model runner。

---

## 小结

Worker 层的设计可以用三句话概括：

1. **生命周期由 executor 的 collective_rpc 驱动**：`init_device → load_model → get_kv_cache_spec → determine_available_memory(profile) → initialize_from_config(分配 KV) → compile_or_warm_up_model(编译+graph) → 循环 execute_model`，顺序固定且每步只做一件事。
2. **GPUModelRunner 用一组持久化的 CPU/GPU 缓冲把"调度结果"翻译成"模型输入"**：`InputBatch` 维护请求槽位与 token 矩阵，`BlockTable` 维护块表并算 slot mapping，`_prepare_inputs` 把它们 gather 成扁平的 `input_ids/positions/seq_lens/query_start_loc`，`set_forward_context` 把注意力元数据喂给模型层；前向后再由 sampler + 簿记逻辑产出 `ModelRunnerOutput`。
3. **性能优化围绕"减少每步同步"展开**：持久化批避免重建、CpuGpuBuffer 非阻塞 H2D、CUDA graph 固定地址、async scheduling 把 D2H 与下一 step 重叠、微批让计算与通信重叠、乐观 spec decode + 延迟校正、CPU 端 gather 而非逐 token 处理。

下一篇 [05 模型执行与层](../models/05_model_executor_layers.md) 将进入 `vllm/model_executor/layers/`，看 `self.model(...)` 内部的线性层、Attention 层、归一化、MoE、RoPE 等如何消费这里准备好的元数据并完成真正的张量计算。
