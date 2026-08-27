# 编译管线

> 源码路径: `vllm/compilation/`

vLLM 用 `torch.compile` + 自定义 Inductor 后端 + CUDA Graph 三件套，把模型 forward 从"逐算子 eager 执行"变成"分块编译 + 形状特化 + 图重放"。这套机制对上层模型完全透明——模型代码里加一个 `@support_torch_compile` 装饰器即可，但底层做了大量工程：

- **一次 Dynamo trace，多次 Inductor 编译**：按"分割算子"（attention、MoE 等不透明 custom op）把 FX 图切成多段，每段针对不同 token 数（decode 的固定 bucket、prefill 的若干尺寸）分别编译；
- **跨进程/跨重启的编译缓存**：用 config hash + 源码 hash + 编译器 hash 决定缓存目录，编译产物（`.so`/图序列化字节）落盘复用；
- **CUDA Graph 模式分层**：PIECEWISE（每段子图各自捕获）、FULL（整图捕获）、FULL_AND_PIECEWISE（v1 默认，decode 用 FULL，prefill 用 PIECEWISE）；
- **编译后 FX pass**：在 Inductor 调度前做算子融合（RMSNorm+量化、attention+量化、all-reduce+RMS、QK-norm+RoPE…）、inplace 化、clone 消除、序列并行等。

> 第 04 篇讲过 model runner 在 `load_model` 里如何用 `CUDAGraphWrapper` 包装模型、在 `capture_model` 里捕获；本篇聚焦"编译"本身——图怎么切、怎么编译、怎么缓存、怎么与 CUDA graph 协作。

---

## 整体结构

```
vllm/compilation/
├── decorators.py            # @support_torch_compile：模型类装饰器，注入 TorchCompileWithNoGuardsWrapper
├── wrapper.py               # TorchCompileWithNoGuardsWrapper：第一次调用时触发 torch.compile 并接管 __call__
├── backends.py              # VllmBackend：torch.compile 的后端；split_graph + CompilerManager + PiecewiseBackend
├── piecewise_backend.py     # PiecewiseBackend：管理一个子图在多个 shape/range 上的编译产物
├── compiler_interface.py    # CompilerInterface + InductorAdaptor/InductorStandaloneAdaptor/EagerAdaptor
├── codegen.py               # 把 split_gm 的拼接逻辑生成为 Python 函数（消除 FX Interpreter 开销）
├── partition_rules.py       # should_split / inductor_partition_rule_context
├── cuda_graph.py            # CUDAGraphWrapper：按 BatchDescriptor 捕获/重放；CUDAGraphStat 指标
├── caching.py               # StandaloneCompiledArtifacts、VllmSerializableFunction（AOT 产物序列化）
├── counter.py               # compilation_counter：编译次数、图数等全局计数器
├── monitor.py               # 运行时监控意外 JIT 编译（延迟尖峰）
├── base_static_graph.py     # 静态图基类
└── passes/
    ├── pass_manager.py           # PostGradPassManager：注册并顺序跑 post-grad pass
    ├── inductor_pass.py          # InductorPass（基类）
    ├── vllm_inductor_pass.py     # VllmInductorPass / VllmPatternMatcherPass / VllmPatternReplacement
    ├── fx_utils.py
    ├── ir/                       # FX 图级 pass
    │   ├── clone_elimination.py
    │   ├── inplace_functionalization.py
    │   └── lowering_pass.py
    ├── utility/                  # 工具 pass
    │   ├── noop_elimination.py / post_cleanup.py
    │   ├── fix_functionalization.py / scatter_split_replace.py
    │   └── split_coalescing.py
    └── fusion/                   # 算子融合 pattern（PatternMatcher）
        ├── rms_quant_fusion.py        # RMSNorm + FP8/NVFP4 量化
        ├── attn_quant_fusion.py       # attention 前后的量化融合
        ├── mla_attn_quant_fusion.py   # MLA 版本
        ├── act_quant_fusion.py        # SiLUAndMul + 量化
        ├── qk_norm_rope_fusion.py     # QK norm + RoPE 融合
        ├── rope_kvcache_fusion.py     # RoPE + KV cache 写入融合
        ├── allreduce_rms_fusion.py    # TP all-reduce + RMSNorm
        ├── collective_fusion.py / sequence_parallelism.py
        ├── minimax_qk_norm_fusion.py  # MiniMax 专用
        ├── rocm_aiter_fusion.py       # ROCm AITER
        └── matcher_utils.py
```

三种编译模式（[config/compilation.py:37](../../vllm/config/compilation.py#L37)）：

| 模式 | 含义 |
|------|------|
| `NONE` (0) | 纯 eager，不编译；只有 CUDA graph（`--enforce-eager` 时连 graph 也关） |
| `STOCK_TORCH_COMPILE` (1) | 原生 `model.compile(fullgraph=True, backend=...)`，不做 vLLM 分段 |
| `VLLM_COMPILE` (3) | vLLM 自定义后端（默认）：分段 + 多形状编译 + 缓存 + 自定义 pass |

---

## 一、入口：`@support_torch_compile`

模型类（如 [LlamaModel](../../vllm/model_executor/models/llama.py#L340)）上的装饰器是整个管线的起点。

### 1.1 装饰器做了什么

[decorators.py:86-331](../../vllm/compilation/decorators.py#L86-L331) 的 `support_torch_compile`（实际实现 `_support_torch_compile`，[L331](../../vllm/compilation/decorators.py#L331)）：

1. **不创建新类**，而是把 `TorchCompileWithNoGuardsWrapper` **动态注入到模型类的基类**里（`cls.__bases__ += (TorchCompileWithNoGuardsWrapper,)`，[L347](../../vllm/compilation/decorators.py#L347)）。这样模型实例同时是 `nn.Module` 和编译包装器，且 MRO 保证 `super().__init__` 正常。
2. **包装 `__init__`**：先调原始 `old_init` 建完模型，再读 `compilation_config`，判断是否需要编译（`do_not_compile`：`mode==NONE/STOCK_TORCH_COMPILE`、`@ignore_torch_compile`、`enable_if` 条件不满足）。
3. 需要编译时，调用 `TorchCompileWithNoGuardsWrapper.__init__(self, ...)` 在实例上挂编译状态。
4. 提供 `dynamic_arg_dims`：声明哪些输入的哪些维度是动态的（通常是 `input_ids: {0: "b"}`、`positions: {0: "b"}`，b 即 token 数）。`_mark_dynamic_inputs`（[L414](../../vllm/compilation/decorators.py#L414)）在首次 trace 时用 `torch._dynamo.mark_dynamic` 标记，让编译图支持符号形状，避免每个 batch 大小都重编译。

`@ignore_torch_compile`（[L58](../../vllm/compilation/decorators.py#L58)）让某个子类跳过编译（如某些模型的视觉塔）。

### 1.2 TorchCompileWithNoGuardsWrapper：首次调用触发编译

[wrapper.py:72](../../vllm/compilation/wrapper.py#L72) 是混入模型的包装器。核心思路是**"no guards"**：

- 普通 `torch.compile` 会为每个不同形状/配置生成 guard，guard 失败就重新 trace，导致大量重编译。
- vLLM 只在**第一次**前向时做一次 Dynamo trace（产出 FX 图），之后用 `VllmBackend` 针对多个具体形状编译 Inductor 产物；运行时由 piecewise backend 按当前 token 数选择已编译的 callable，**不再走 Dynamo guard 判定**。

`__init__` 里（[wrapper.py:72-160](../../vllm/compilation/wrapper.py#L72-L160)）准备 `torch.compile` 的配置（动态维度、fullgraph、backend=`VllmBackend`）。`__call__`（[wrapper.py:169](../../vllm/compilation/wrapper.py#L169)）在第一次真实调用时触发编译并替换自己的 forward，后续直接走编译产物；`_dispatch_to_compiled_code`（[L265](../../vllm/compilation/wrapper.py#L265)）处理字节码 hook（让模型的 `forward` 调用被重定向到编译版本，而不需要改模型调用点）。

`reset_compile_wrapper(model)`（[wrapper.py:285](../../vllm/compilation/wrapper.py#L285)）在测试/重载时重置。

---

## 二、VllmBackend：切图与编排

[backends.py:800](../../vllm/compilation/backends.py#L800) 是 `torch.compile(backend=...)` 指定的后端。Dynamo  trace 出一张完整 FX 图后会调用 `VllmBackend.__call__(graph, example_inputs)`。

### 2.1 `__call__` 主流程

[backends.py:1015](../../vllm/compilation/backends.py#L1015) 编排：

1. **算缓存 key**：收集环境因子（`envs.compile_factors()`：torch 版本、GPU 架构、关键 env）、`vllm_config.compute_hash()`、被 trace 的源码文件内容 hash、编译器 hash，组合成 `hash_key`，定位缓存目录 `~/.cache/vllm/torch_compile_cache/<hash_key>/rank_<rank>_<dp_rank>/<prefix>`（[L1058-1074](../../vllm/compilation/backends.py#L1058-L1074)）。prefix 区分 backbone/eagle_head/encoder。
2. **初始化 CompilerManager**（[L1094](../../vllm/compilation/backends.py#L1094)）：加载已有缓存或准备编译。
3. **配置 post-grad passes**：`self.pass_manager.configure(vllm_config)`（[L929](../../vllm/compilation/backends.py#L929)）把所有融合/工具 pass 注册进 Inductor 的 `post_grad_custom_post_pass`。
4. **切图**：`split_graph(graph, compilation_config.splitting_ops)`（[L548](../../vllm/compilation/backends.py#L548)）。
5. 对每个子图创建 `PiecewiseBackend`，按需用 `wrap_with_cudagraph_if_needed`（[L628](../../vllm/compilation/backends.py#L628)）包一层 CUDA graph。
6. 用 `codegen` 生成拼接函数，作为返回的可调用对象。

### 2.2 split_graph：在不透明算子处切分

[backends.py:548-622](../../vllm/compilation/backends.py#L548-L622)：

- `splitting_ops` 来自配置，通常是 `torch.ops.vllm.unified_attention_with_output`、`unified_kv_cache_update`、MoE 算子、mamba 算子等——它们内部走自定义 kernel，不需要/不适合被 Inductor 编译，且边界天然。
- 遍历 FX 节点，遇到 splitting op 就新建一个子图 id（连续的 splitting op 保持在同一子图）；`getitem` 跟随其来源节点，避免把整个 tuple 传过子图边界（[L565-571](../../vllm/compilation/backends.py#L565-L571)）。
- 用 PyTorch 的 `split_module` 按 `node_to_subgraph_id` 切成 `submod_0, submod_1, ...`，保留原始顺序（对有 inplace mutation 的图很重要）。
- 产出 `split_gm`（拼接图模块）和 `list[SplitItem]`（每个子图名、id、是否以 splitting op 开头）。

`_decompose_size_nodes`（[L479](../../vllm/compilation/backends.py#L479)）先把图里的 size 计算节点拆出来，确保符号形状在子图间正确传播。

### 2.3 为什么要切图

- Attention/MoE 是大块自定义 kernel，Inductor 看到的是 opaque op，把它们之间的"小算子串"（norm、roate、量化、残差、加 bias）单独编译，Inductor 才能做融合和优化；
- 每段可以用**不同的编译策略/cudagraph 模式**：attention 段可能走 PIECEWISE cudagraph，外围计算段也可以；
- 分段后每段的形状特化更精细，且可以单独缓存。

### 2.4 codegen：消除 Interpreter 开销

[codegen.py](../../vllm/compilation/codegen.py) 把 `split_gm` 原本通过 `fx.Interpreter` 逐节点调度子模块的逻辑，**生成成一个纯 Python 函数源码**（`generate_execution_code_with_name`）：内联子模块调用、做生命周期分析（及时释放张量）、把常量提出来。这样运行时就是普通的 Python 函数调用链，没有 `nn.Module.__call__` 和 FX dispatch 的开销。生成的函数就是最终返回给 Dynamo 的 compiled callable。

---

## 三、PiecewiseBackend：一个子图的多形状编译

[piecewise_backend.py:86](../../vllm/compilation/piecewise_backend.py#L86) 包装一个子图（submodule），负责把它在**多个 token 数范围**上编译出多个版本。

### 3.1 构造与范围

构造时（[L87](../../vllm/compilation/piecewise_backend.py#L87)）拿到：
- 子图的 FX graph、`compiler_config`、`compile_ranges`（如 `[Range(1,1), Range(2,4), Range(5,8), ...]`，对应 cudagraph bucket 与编译尺寸）；
- 是否是第一/最后一个子图（决定能否包 FULL cudagraph）；
- 符号形状索引 `sym_shape_indices`（哪个输入维度是动态的 b）。

`RangeEntry` 记录一个编译范围对应的编译 callable/handle。

### 3.2 编译所有范围

`compile_all_ranges`（[L245](../../vllm/compilation/piecewise_backend.py#L245)）对每个 range：
- 用 `create_concrete_args(graph, size)`（[L37](../../vllm/compilation/piecewise_backend.py#L37)）把符号 b 具象成具体 size 造假输入；
- 调 `CompilerManager.compile(graph, example_inputs, config, range, key)` 得到编译产物；
- 如果启用 cudagraph，用 `CUDAGraphWrapper` 包装该子图（PIECEWISE 模式）；
- 记录 `RangeEntry`。

`load_all_ranges`（[L319](../../vllm/compilation/piecewise_backend.py#L319)）在缓存命中时直接从 handle 加载，不重新编译。

### 3.3 运行时分发

`__call__`（[L358](../../vllm/compilation/piecewise_backend.py#L358)）取当前运行时形状（token 数），用 `_find_range_for_shape`（[L343](../../vllm/compilation/piecewise_backend.py#L343)）找到覆盖它的最窄已编译 range，调用对应的 callable；找不到就走 eager/一般编译版本。这样 decode（固定 1+num_spec token）命中精确 bucket，prefill 命中较大的 range，避免每次都重新编译。

`to_bytes`/`from_bytes`（[L209](../../vllm/compilation/piecewise_backend.py#L209)）把各 range 的独立编译产物序列化，供 mega AOT artifact 收集（[backends.py:867](../../vllm/compilation/backends.py#L867) `collect_standalone_compile_artifacts`）。

---

## 四、CompilerInterface 与 Inductor 适配

[compiler_interface.py](../../vllm/compilation/compiler_interface.py) 抽象"编译器"，有三个实现：

| 类 | 作用 |
|----|------|
| `EagerAdaptor`（[L768](../../vllm/compilation/compiler_interface.py#L768)） | 不编译，直接返回原 graph（降级/调试用） |
| `InductorAdaptor`（[L449](../../vllm/compilation/compiler_interface.py#L449)） | 进程内调用 `torch._inductor.compile_fx`，劫持 hash 与文件路径以支持缓存 |
| `InductorStandaloneAdaptor`（[L251](../../vllm/compilation/compiler_interface.py#L251)） | PyTorch 2.8+ 的 standalone_compile API，支持把编译产物保存成可独立加载的 artifact（mega AOT） |

### 4.1 InductorAdaptor.compile 的关键技巧

[compiler_interface.py:482-658](../../vllm/compilation/compiler_interface.py#L482-L658)：

1. 深拷贝 graph（Inductor 会原地改图）；
2. 用 `ExitStack` 给 PyTorch 打一堆 monkey patch：
   - `compiled_fx_graph_hash` 被劫持以拿到图的缓存 hash；
   - `FxGraphCache._get_shape_env` 返回 `AlwaysHitShapeEnv`（[L117](../../vllm/compilation/compiler_interface.py#L117)）——因为 vLLM 在 Dynamo tracing 上下文之外多次调 Inductor，没有 shape env 会让 code cache 查找失败，这个假 env 让 guard 总是命中；
   - `_check_can_cache` 被改成 no-op，强制可缓存（Inductor 默认对高阶 op/非 tracing 上下文拒绝缓存）；
   - 关闭 remote cache 和 autograd cache；
   - 临时清空 `TracingContext`，避免子图的 FakeTensorMode 与外层 Dynamo 的冲突（[L620-635](../../vllm/compilation/compiler_interface.py#L620-L635)）。
3. 调 `compile_fx(graph, example_inputs, inner_compile=hijacked_compile_fx_inner, ...)`，从闭包里取出 `hash_str` 和编译产物的 `file_path` 作为 handle 返回。

`AlwaysHitShapeEnv` 的注释（[L119-140](../../vllm/compilation/compiler_interface.py#L119-L140)）解释了核心问题：**一次 Dynamo bytecode 编译，多次 Inductor 编译**——正常 torch.compile 假设两者一一对应，vLLM 打破了这个假设，需要骗过 Inductor 的缓存守卫。

### 4.2 CompilerManager

[backends.py:124](../../vllm/compilation/backends.py#L124) 管理多个子图共用的编译器实例：`compute_hash`（把编译器版本/pass 列表纳入缓存 key）、`initialize_cache`（把 Inductor 缓存目录指向 local_cache_dir）、`compile`（按 range 调对应 adaptor）、`load`/`save_to_file`（缓存持久化）、`compile_context`（设置 inductor config 的上下文）。

`make_compiler`（[backends.py:96](../../vllm/compilation/backends.py#L96)）根据配置选择 Inductor/Standalone/Eager。

---

## 五、CUDA Graph 协作

### 5.1 两种图模式与包装位置

回顾第 04 篇：model runner 在 `load_model` 时用 `CUDAGraphWrapper` 包装**整个模型**（FULL 模式）或 `UBatchWrapper`。除此之外，编译期还有一层：

- **PIECEWISE 图**：`wrap_with_cudagraph_if_needed`（[backends.py:628](../../vllm/compilation/backends.py#L628)）在每个 `PiecewiseBackend` 外面包一个 `CUDAGraphWrapper(runtime_mode=PIECEWISE)`。这样单个编译子段可以独立捕获/重放，适合 prefill（形状变化大、不一定命中整图 bucket）。
- **FULL 图**：model runner 层的 `CUDAGraphWrapper(runtime_mode=FULL)` 包住整个 split_gm（或经过 codegen 的拼接函数），捕获端到端前向。

`CUDAGraphMode`（[config/compilation.py:53](../../vllm/config/compilation.py#L53)）：
- `NONE`：不捕获；
- `PIECEWISE`：只捕 piecewise 子段；
- `FULL`：只捕整图；
- `FULL_AND_PIECEWISE`（v1 默认）：decode 用 FULL，prefill/mixed 用 PIECEWISE。

### 5.2 CUDAGraphWrapper 的运行时分发

[cuda_graph.py:145](../../vllm/compilation/cuda_graph.py#L145)：

- 持有 `concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry]`（[L207](../../vllm/compilation/cuda_graph.py#L207)），key 是 `BatchDescriptor`（token 数、是否 uniform decode、lora 数等）。
- `__call__`（[L233](../../vllm/compilation/cuda_graph.py#L233)）从 `get_forward_context()` 取 `batch_descriptor` 和 `cudagraph_runtime_mode`：
  - runtime mode 是 NONE 或与自己的 mode 不匹配 → 直接调 `runnable`（eager/编译版）；
  - 否则按 batch descriptor 查：命中则 **replay** 对应 graph；未命中则当场**捕获**一条新 graph 存入字典再 replay。
- 通过 `__getattr__` 把属性访问转发给底层 runnable，因此它对模型代码透明（`model.layers`、`model.norm` 仍可访问）。
- 不负责把输入拷进/拷出持久缓冲——那是 model runner 的责任（用第 04 篇那些预分配的 `input_ids/positions/...` 张量，地址固定才能被 graph 捕获）。

所有实例登记在 `_all_instances`（WeakSet，[L170](../../vllm/compilation/cuda_graph.py#L170)），profile cudagraph 内存时可统一替换 graph pool 并清理。

### 5.3 CudagraphDispatcher

[v1/cudagraph_dispatcher.py:15](../../vllm/v1/cudagraph_dispatcher.py#L15) 在 model runner 里维护 `cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]]`（[L44](../../vllm/v1/cudagraph_dispatcher.py#L44)）。`initialize_cudagraph_keys` 根据编译配置的 capture sizes、uniform decode、lora 组合等生成所有需要捕获的 `BatchDescriptor`；`get_capture_descs`（[L325](../../vllm/v1/cudagraph_dispatcher.py#L325)）返回按 mode 分组的捕获列表（第 04 篇 `capture_model` 遍历它）。运行时 wrapper 用 `batch_descriptor in cudagraph_keys[mode]` 判断是否可用图。

### 5.4 CUDAGraphStat 与监控

[cuda_graph.py:33](../../vllm/compilation/cuda_graph.py#L33) 的 `CUDAGraphStat` 记录一次前向用了哪个 graph（new/eager/piecewise/full）。`CUDAGraphLogging`（[L40](../../vllm/compilation/cuda_graph.py#L40)）聚合并 `generate_metric_table` 打印命中率，帮助判断 cudagraph 配置是否合理。[monitor.py](../../vllm/compilation/monitor.py) 在 warmup 后激活，监控运行时意外的 Inductor/JIT 编译（会导致延迟尖峰）并报警。

---

## 六、编译缓存

### 6.1 缓存 key 与目录

如第二节所述，缓存目录由四类因子决定：
- **env_hash**：torch 版本、CUDA、GPU 架构、影响 kernel 的 env；
- **config_hash**：`VllmConfig.compute_hash()`（模型/并行/量化/编译配置）；
- **code_hash**：被 `torch.compile` trace 到的源码文件内容 SHA256（[backends.py:1037-1050](../../vllm/compilation/backends.py#L1037-L1050)），改了模型代码自动失效；
- **compiler_hash**：编译器/pass 自身版本。

`cache_key_factors.json`（[L1116](../../vllm/compilation/backends.py#L1116)）把原始因子落盘便于调试。

### 6.2 InductorAdaptor 的缓存复用

`InductorAdaptor.initialize_cache`（[compiler_interface.py:463](../../vllm/compilation/compiler_interface.py#L463)）把 PyTorch Inductor 的缓存目录（`TORCHINDUCTOR_CACHE`/fbcode cache 等）重定向到 vLLM 的 local_cache_dir。这样相同 hash 下，第二次启动直接命中 Inductor 编译好的 `.so`，`load`（[L660](../../vllm/compilation/compiler_interface.py#L660)）从 handle `(hash_str, file_path)` 重新加载 compiled callable，完全跳过编译。

### 6.3 Standalone / Mega AOT artifact

[caching.py](../../vllm/compilation/caching.py) 提供更进一步的离线编译能力：
- `VllmSerializableFunction`（[L166](../../vllm/compilation/caching.py#L166)）把 FX GraphModule 序列化为字节（`serialize_graph_module`），并能序列化/反序列化整个 compile artifacts（[L252](../../vllm/compilation/caching.py#L252)、[L300](../../vllm/compilation/caching.py#L300)）。
- `StandaloneCompiledArtifacts`（[L37](../../vllm/compilation/caching.py#L37)）按 `submod_name → shape → bytes` 存各 piecewise 子图各形状的独立产物。
- 开启 `VLLM_USE_MEGA_AOT_ARTIFACT` 时，`VllmBackend.collect_standalone_compile_artifacts`（[backends.py:867](../../vllm/compilation/backends.py#L867)）把所有子图产物收集成一个 mega artifact，可在**没有编译环境的机器上加载运行**（AOT 部署）。
- `aot_compile_hash_factors`/`_compute_code_hash`（[caching.py:565](../../vllm/compilation/caching.py#L565)）计算 AOT 缓存 hash。

`_patch_standalone_compile_atomic_save`（[compiler_interface.py:210](../../vllm/compilation/compiler_interface.py#L210)）修复并发写缓存的原子性问题。

---

## 七、编译 Pass 系统

### 7.1 PostGradPassManager

[passes/pass_manager.py:80](../../vllm/compilation/passes/pass_manager.py#L80) 是 post-grad pass 的统一入口（继承自 PyTorch `CustomGraphPass`）。`configure(config)`（[L132](../../vllm/compilation/passes/pass_manager.py#L132)）按当前 vLLM 配置/平台实例化并 `add()` 所有 pass；`__call__(graph)` 顺序执行并通过 `uuid()` 把 pass 列表纳入编译缓存 key（pass 变了缓存失效）。

pass 分两类：

- **FX 图级 pass**（`passes/ir/`、`passes/utility/`）：直接操作 FX Graph。
- **Inductor IR 级 pass**（`passes/fusion/`，继承 `VllmInductorPass`）：在 Inductor lowering 后的 IR 上用 PatternMatcher 做融合。

### 7.2 VllmInductorPass 与 PatternReplacement

[passes/vllm_inductor_pass.py](../../vllm/compilation/passes/vllm_inductor_pass.py)：

- `VllmInductorPass`（[L34](../../vllm/compilation/passes/vllm_inductor_pass.py#L34)）：基类，提供 `time_and_log`、`dump_graph`、`begin/end_and_log`，每个 pass 有 `uuid`。
- `VllmPatternMatcherPass`（[L92](../../vllm/compilation/passes/vllm_inductor_pass.py#L92)）：封装 Inductor 的 `PatternMatcherPass`，注册一组 search-replace pattern。
- `VllmPatternReplacement`（[P, R]`，[L194](../../vllm/compilation/passes/vllm_inductor_pass.py#L194)）：声明 `pattern()`（要匹配的计算子图）和 `replacement()`（替换成的融合算子），用 `empty/empty_bf16/...` 造假输入。
- `VllmFusionPatternMatcherPass`（[L266](../../vllm/compilation/passes/vllm_inductor_pass.py#L266)）：批量注册多个 `VllmPatternReplacement`，`__call__` 时在图上 apply 并 log 匹配数。

### 7.3 主要融合 pass

| Pass | 融合内容 |
|------|---------|
| `rms_quant_fusion` | RMSNorm 输出 → FP8/NVFP4 量化合并成一个 kernel |
| `attn_quant_fusion` | attention 前后的动态量化/反量化与 attention op 融合 |
| `mla_attn_quant_fusion` | MLA 场景的量化融合（DeepSeek） |
| `act_quant_fusion` | `SiluAndMul`/`GeluAndMul` 等激活 + 量化 |
| `qk_norm_rope_fusion` | Q/K RMSNorm + RoPE 融合（如 MiniMax、Qwen3） |
| `rope_kvcache_fusion` | RoPE 之后的 KV cache 写入融合 |
| `allreduce_rms_fusion` | TP 的 all-reduce + 后续 RMSNorm 融合（减少一次显存往返） |
| `collective_fusion` / `sequence_parallelism` | 集合通信融合与序列并行的 all-gather/reduce-scatter |
| `minimax_qk_norm_fusion` | MiniMax 专用 QK-norm 融合 |
| `rocm_aiter_fusion` | ROCm AITER 库的融合 |

匹配工具在 [fusion/matcher_utils.py](../../vllm/compilation/passes/fusion/matcher_utils.py)（如 `MatcherQuantFP8`）。

### 7.4 图级工具 pass

- `ir/clone_elimination.py`：去掉冗余 `aten.clone`；
- `ir/inplace_functionalization.py`：把可安全原地的算子转 inplace（省显存）；
- `ir/lowering_pass.py`：自定义 lowering；
- `utility/noop_elimination.py`：消除 no-op；
- `utility/fix_functionalization.py`：修复 functionalization 后的问题；
- `utility/scatter_split_replace.py`、`split_coalescing.py`：scatter/split 优化；
- `utility/post_cleanup.py`：最终清理。

这些 pass 让编译后的图在 Inductor 调度前就尽量融合、减少中间张量和 kernel 启动。

---

## 八、计数器与可观测性

- **[counter.py](../../vllm/compilation/counter.py)**：`compilation_counter` 全局对象，记录 `num_models_seen`、`num_inductor_compiles`、`num_cudagraph_captured`、`num_gpu_runner_capture_triggers` 等，编译日志和统计用到。
- [monitor.py](../../vllm/compilation/monitor.py)：warmup 结束后 `activate()`，hook 进 Inductor 编译入口，一旦在稳态推理中触发编译就记录警告（意外编译意味着延迟尖峰，通常是遇到没覆盖的形状）。
- [base_static_graph.py](../../vllm/compilation/base_static_graph.py)：静态图基类，为需要固定结构的子模块（如某些 cudagraph 友好组件）提供支持。

---

## 九、端到端串起来

把第 04、06 篇与本篇连起来，一次完整的"编译+图捕获"时间线：

```
引擎启动
  │
  Worker.init_device → 构造 GPUModelRunner
  Worker.load_model
  ├─ 模型 __init__：@support_torch_compile 把 TorchCompileWithNoGuardsWrapper 混入
  │                 （此时还没编译，do_not_compile 检查）
  ├─ mode=STOCK_TORCH_COMPILE：直接 model.compile(fullgraph=True)，结束
  └─ mode=VLLM_COMPILE：用 CUDAGraphWrapper(FULL) 包装模型（加载模型后）

determine_available_memory → profile_run → _dummy_run(max_num_tokens)
  │  第一次前向触发 TorchCompileWithNoGuardsWrapper：
  │   1. Dynamo trace 一次（mark_dynamic b 维度）→ 完整 FX graph
  │   2. torch.compile 调 VllmBackend.__call__：
  │      - 算缓存 hash、建/加载缓存目录
  │      - PostGradPassManager.configure() 注册所有融合 pass
  │      - split_graph 在 attention/MoE 处切分
  │      - 每段 → PiecewiseBackend（compile_ranges 各尺寸）
  │      - 每段 InductorAdaptor.compile（打 patch、compile_fx、得 hash+file）
  │      - 按需包 PIECEWISE CUDAGraphWrapper
  │      - codegen 生成拼接 Python 函数
  │   3. 返回 compiled callable，后续前向走它

compile_or_warm_up_model
  ├─ 对 compile_sizes / cudagraph_capture_sizes 逐个 _dummy_run
  │     → PiecewiseBackend 为每个 size 编译/命中缓存
  ├─ kernel_warmup（Triton JIT）
  └─ capture_model：CudagraphDispatcher.get_capture_descs 给出 BatchDescriptor 列表
        → CUDAGraphWrapper 按 desc 捕获 FULL/PIECEWISE graph
        → lock_workspace()

稳态执行
  GPUModelRunner.execute_model
   → set_forward_context(batch_descriptor, cudagraph_runtime_mode)
   → model(...) → CUDAGraphWrapper.__call__
        ├─ mode/batch 命中 → graph.replay()
        └─ 未命中 → runnable()（编译版/eager），必要时现场捕获
```

缓存命中时，第 2 步的 Inductor 编译全部跳过，直接从 `.so`/artifact 加载，显著缩短启动时间。

---

## 小结

vLLM 的编译管线可以概括为四句话：

1. **`@support_torch_compile` 通过混入 wrapper 在首次前向时触发一次 Dynamo trace**，用动态维度避免按形状重编译；
2. **`VllmBackend` 在 attention/MoE 等不透明算子处把图切成多段**，每段交给 `PiecewiseBackend` 针对多个 token 数范围分别用 Inductor 编译，再用 `codegen` 生成低开销拼接函数；
3. **`CompilerInterface`/`InductorAdaptor` 用一组 monkey patch 实现"一次 trace、多次 Inductor 编译"和进程/跨重启缓存**，key 由 env+config+源码+编译器 hash 决定，mega AOT 支持离线产物部署；
4. **CUDA graph 分 PIECEWISE（子段）和 FULL（整图）两层**，`CUDAGraphWrapper` 按 `BatchDescriptor` 捕获/重放，与编译产物正交协作；`PostGradPassManager` 在 Inductor IR 上做大量融合（norm+量化、allreduce+norm、QK-norm+RoPE…）进一步提速。

下一篇 [08 分布式推理](../distributed/08_distributed.md) 将进入 `vllm/distributed/`，看 TP/PP/EP/DP/CP 等并行策略、通信组、all-to-all 与 KV transfer 如何把这里的单卡前向扩展到多卡/多节点。
