# 注意力后端

> 源码路径: `vllm/v1/attention/`

注意力后端（Attention Backend）是 vLLM 与具体注意力计算库（FlashAttention、FlashInfer、Triton、CUTLASS/FlashMLA 等）之间的抽象层。它解决三个问题：

1. **KV cache 的物理布局是什么**（张量形状、维度顺序、量化方式）；
2. **每一步前向时，如何把调度器给出的"请求 → token 数/block table"翻译成底层 kernel 能理解的元数据**（metadata builder）；
3. **真正执行注意力计算**（impl 的 forward），并处理 prefill/decode、分页 KV、cascade attention、滑动窗口、量化、context parallelism 等。

本模块是一个"注册-选择-构建-执行"的插件体系：模型层只依赖抽象基类 `AttentionBackend`，具体用哪个 kernel 在启动时根据硬件能力、模型结构（是否 MLA、是否稀疏）、KV cache dtype 等自动选定，也可由 `--attention-backend` 强制指定。

---

## 整体结构

```
vllm/v1/attention/
├── backend.py          # 抽象基类：AttentionBackend / AttentionImpl / AttentionMetadataBuilder + CommonAttentionMetadata
├── selector.py         # get_attn_backend() 按平台+配置选择后端（带缓存）
├── backends/
│   ├── registry.py     # AttentionBackendEnum / MambaAttentionBackendEnum：所有后端的 qualname 注册表
│   ├── flash_attn.py   # FlashAttentionBackend（参考实现，Hopper FA3）
│   ├── flashinfer.py   # FlashInferBackend（Blackwell，含 TRT-LLM gen kernel）
│   ├── triton_attn.py  # TritonAttentionBackend（可移植，统一 prefill/decode kernel）
│   ├── flex_attention.py  # 基于 PyTorch flex_attention（研究/自定义 mask）
│   ├── flash_attn_diffkv.py  # K/V head_dim 不同
│   ├── cpu_attn.py / rocm_*.py / turboquant_attn.py / ...  # 平台/专用后端
│   ├── mamba_attn.py / mamba1_attn.py / mamba2_attn.py     # SSM 后端基类与实现
│   ├── linear_attn.py / gdn_attn.py / short_conv_attn.py   # 线性注意力/SSM 变体
│   └── mla/            # MLA（DeepSeek V2/V3/V4）专用后端
│       ├── flashattn_mla.py / flashmla.py / flashinfer_mla.py / triton_mla.py / cutlass_mla.py / ...
│       ├── flashmla_sparse.py / flashinfer_mla_sparse.py / ...（稀疏 MLA）
│       ├── indexer.py / sparse_swa.py / sparse_utils.py / compressor_utils.py
│       └── prefill/    # MLA prefill 子注册表（与 decode 解耦）
└── ops/                # 低层 Triton/C++ 算子（被各 backend 调用）
    ├── triton_unified_attention.py / triton_decode_attention.py / triton_prefill_attention.py
    ├── prefix_prefill.py / chunked_prefill_paged_decode.py / merge_attn_states.py
    ├── triton_reshape_and_cache_flash.py / paged_attn.py
    ├── common.py（CP LSE 修正）/ dcp_alltoall.py（DCP all-to-all）
    └── deepseek_v4_ops/  # DeepSeek V4 融合算子
```

调用分层：

```
模型层 Attention.forward (model_executor/layers/attention/attention.py)
   │  torch.ops.vllm.unified_attention_with_output
   ▼
unified_attention_with_output()  →  self.impl.forward(layer, q, k, v, kv_cache, attn_metadata, output)
   │
   ▼
AttentionImpl 子类（FlashAttentionImpl / FlashInferImpl / TritonAttentionImpl / MLAAttentionImpl ...）
   │  使用 builder 产出的、每层专属的 AttentionMetadata
   │  调用 ops/ 下的 Triton kernel 或 _custom_ops / 第三方库
   ▼
分页 KV cache 张量（由 model runner 按 backend.get_kv_cache_shape() 分配）
```

三类核心对象的分工（都在 [backend.py](../../vllm/v1/attention/backend.py) 定义）：
- **`AttentionBackend`**（静态工厂）：声明能力（支持的 dtype/head_size/block_size/MLA/稀疏/sink…）、KV cache 形状、产出 Impl 类和 Builder 类；
- **`AttentionMetadataBuilder`**（每 KV group 一个实例）：把跨层共享的 `CommonAttentionMetadata` 转换为某层专属的 metadata；
- **`AttentionImpl`**（每层一个实例）：持有 scale/sliding_window 等参数，执行真正的 forward。

---

## 一、AttentionType 与后端能力声明

### 1.1 AttentionType

[backend.py:32-45](../../vllm/v1/attention/backend.py#L32-L45) 用字符串枚举（为兼容 `torch.compile`）：

| 类型 | 含义 |
|------|------|
| `DECODER` | decoder-only 自注意力（默认） |
| `ENCODER` | encoder-decoder 中 encoder 的自注意力 |
| `ENCODER_ONLY` | 纯 encoder 模型（如 ViT）的双向注意力 |
| `ENCODER_DECODER` | 跨注意力（decoder Q 对 encoder K/V） |

默认后端只支持 `DECODER`，需覆盖 `supports_attn_type` 才能用于其他类型。

### 1.2 AttentionBackend 的能力方法

[backend.py:55-343](../../vllm/v1/attention/backend.py#L55-L343) 是抽象基类。核心静态/类方法：

| 方法 | 作用 |
|------|------|
| `get_name()` | 后端名（对应枚举） |
| `get_impl_cls()` / `get_builder_cls()` | 产出 Impl / Builder 类 |
| `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, ...)` | **KV cache 张量形状**，决定物理布局 |
| `get_kv_cache_stride_order()` | 物理内存维度排列（可与逻辑 shape 不同，如 HND） |
| `get_supported_kernel_block_sizes()` | kernel 要求的 block_size（返回 `int` 或 `MultipleOf`） |
| `get_kv_cache_block_dim()` | 用哨兵值探测哪个维度是 block 维 |
| `get_required_kv_cache_layout()` | 要求全局 KV cache layout（`NHD`/`HND`），见下文 |
| `is_mla()` / `is_sparse()` / `is_ssm()` | 是否 MLA / 稀疏 / 状态空间模型 |
| `supports_sink()` / `supports_mm_prefix()` / `supports_non_causal()` / `supports_alibi_sqrt()` / `supports_per_head_quant_scales()` / `supports_batch_invariance()` | 各特性开关 |
| `supports_dtype/head_size/kv_cache_dtype/block_size/attn_type/compute_capability()` | 单项能力检查 |
| `supports_combination(...)` | 组合约束（某些特性不能共存），返回不可用原因字符串 |
| `validate_configuration(...)` | 汇总所有检查，返回 `invalid_reasons` 列表 |

关键类属性：
- `supported_dtypes`（默认 fp16/bf16）、`supported_kv_cache_dtypes`（默认 auto/fp16/bf16）；
- **`forward_includes_kv_cache_update`**（[backend.py:66](../../vllm/v1/attention/backend.py#L66)）：后端 forward 是否内部完成 KV cache 写入。默认为 True；若为 False（如 TurboQuant），模型层会在 attention 前显式调用 `unified_kv_cache_update`。

`validate_configuration` 是后端选择时的统一裁判：选择器遍历候选后端，第一个返回空 `invalid_reasons` 的胜出。

### 1.3 CUDA Graph 支持等级

[backend.py:499-513](../../vllm/v1/attention/backend.py#L499-L513)：

```python
class AttentionCGSupport(Enum):
    ALWAYS = 3                    # 总是支持，含 mixed prefill-decode
    UNIFORM_BATCH = 2            # batch 内 query 长度相同（spec decode）
    UNIFORM_SINGLE_TOKEN_DECODE = 1  # 仅纯单 token decode
    NEVER = 0
```

Builder 通过 `_cudagraph_support` 类属性和 `get_cudagraph_support()` 声明自己的等级，model runner 据此决定能否捕获 CUDA graph。

---

## 二、后端注册与选择

### 2.1 注册表 `backends/registry.py`

两个枚举把后端名映射到实现类的全限定名字符串，**懒加载**以避免导入可选依赖：

- `AttentionBackendEnum`（[registry.py:34-90](../../vllm/v1/attention/backends/registry.py#L34-L90)）：`FLASH_ATTN`、`FLASHINFER`、`TRITON_ATTN`、`FLASHMLA`、`FLASH_ATTN_MLA`、`FLASHINFER_MLA`、`CUTLASS_MLA`、`TRITON_MLA`、`FLASHMLA_SPARSE`、`FLASHINFER_MLA_SPARSE`、`ROCM_*`、`CPU_ATTN`、`TURBOQUANT`、`FLEX_ATTENTION`、`CUSTOM` 等；
- `MambaAttentionBackendEnum`（[registry.py:136-153](../../vllm/v1/attention/backends/registry.py#L136-L153)）：`MAMBA1`、`MAMBA2`、`SHORT_CONV`、`LINEAR`、`GDN_ATTN`。

每个枚举成员提供 `get_path()` / `get_class()`（`resolve_obj_by_qualname` 动态导入）和 `is_overridden()`。`register_backend()` 装饰器（[registry.py:203-255](../../vllm/v1/attention/backends/registry.py#L203-L255)）允许第三方或测试用 `_ATTN_OVERRIDES` 字典在运行时替换/注册实现。`CUSTOM = None` 是第三方后端的占位符，注册前不能使用。

### 2.2 选择器 `selector.py`

`get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla, has_sink, use_sparse, use_mm_prefix, use_per_head_quant_scales, attn_type, num_heads)`（[selector.py:52-102](../../vllm/v1/attention/selector.py#L52-L102)）：

1. 把所有选择因子打包成可哈希的 `AttentionSelectorConfig`（NamedTuple）；
2. 取当前 vllm_config 的 `attention_config.backend`（用户强制项）和 `block_size`；
3. 调用带 `@cache` 的 `_cached_get_attn_backend`（[selector.py:105-136](../../vllm/v1/attention/selector.py#L105-L136)）：
   - 委托 **平台对象** `current_platform.get_attn_backend_cls(backend, attn_selector_config, num_heads)` 返回候选类的 qualname；
   - `resolve_obj_by_qualname` 导入；
   - 若后端声明了 `get_required_kv_cache_layout()`，调用 `set_kv_cache_layout()` 设置全局 KV cache layout。

平台对象（`vllm/platforms/cuda.py`、`rocm.py`、`xpu.py`、`cpu.py`）持有**按硬件代际和模型特性排序的优先级列表**，逐个调用 `validate_configuration`，第一个通过的胜出。例如 CUDA 上：
- **Hopper（SM90）稠密 MLA**：`FLASH_ATTN_MLA`（FA3）优先，其次 `FLASHMLA`、`TRITON_MLA`；
- **Blackwell（SM100）稠密 MLA**：`FLASHINFER_MLA` 优先，其次 `TOKENSPEED_MLA`、`CUTLASS_MLA`、`FLASHMLA`、`TRITON_MLA`；
- 标准注意力：FlashAttention / FlashInfer / Triton 按能力与配置选择。

`get_mamba_attn_backend(mamba_type)`（[selector.py:139-158](../../vllm/v1/attention/selector.py#L139-L158)）为 SSM 选择后端，并校验 `VLLM_BATCH_INVARIANT` 兼容性。

### 2.3 KV cache layout：NHD vs HND

[backends/utils.py:41-85](../../vllm/v1/attention/backends/utils.py#L41-L85) 定义全局 layout 开关：

- `KVCacheLayoutType = Literal["NHD", "HND"]`；
- `get_kv_cache_layout()` 带 `lru_cache`，优先级：后端 `set_kv_cache_layout()` 强制 > `VLLM_KV_CACHE_LAYOUT` 环境变量 > KV connector 要求；
- NHD = `[num_blocks, block_size, num_kv_heads, head_dim]`（多数后端）；
- HND = `[num_blocks, num_kv_heads, block_size, head_dim]`，FlashInfer 的 TRT-LLM gen kernel 等需要此布局（通过 `get_kv_cache_stride_order` 物理重排）。

`PAD_SLOT_ID = -1`、`NULL_BLOCK_ID = 0`（[utils.py:44-45](../../vllm/v1/attention/backends/utils.py#L44-L45)）是两个重要常量：padding token 的 slot 映射为 -1（kernel 跳过写入），block 0 是永久保留的 null block（与 KV Cache 章节一致）。

---

## 三、CommonAttentionMetadata：跨层共享的元数据

[backend.py:352-493](../../vllm/v1/attention/backend.py#L352-L493) 是 model runner 在每步前向**只构建一次**、所有注意力层共享的元数据。理解字段是理解后端的关键：

| 字段 | 形状/类型 | 含义 |
|------|-----------|------|
| `query_start_loc` / `query_start_loc_cpu` | `(batch+1,)` | 每个请求的 query 在拼平 token 张量中的起止偏移（前缀和） |
| `seq_lens` | `(batch,)` | 每个请求的**已计算总长度**（context_len + query_len） |
| `num_reqs` / `num_actual_tokens` | int | 请求数 / batch 中真实 token 数（去 padding） |
| `max_query_len` / `max_seq_len` | int | 最长 query / 最长上下文（可能是上界） |
| `block_table_tensor` | `(batch, max_blocks)` | 每个请求的逻辑块→物理块映射 |
| `slot_mapping` | `(num_tokens,)` | 每个 token 的物理 KV 槽位（`block_id*block_size + offset`） |
| `causal` | bool | 是否因果掩码 |
| `positions` | `(num_tokens,)` | token 位置（可选，用于预计算位置相关元数据） |
| `is_prefilling` | `(batch,) bool` | 请求是否仍在 prefill 阶段 |
| `seq_lens_cpu_upper_bound` | `(batch,)` | seq_lens 的 CPU 上界（async spec decode 优化用） |
| `encoder_seq_lens` / `encoder_seq_lens_cpu` | | encoder-decoder 跨注意力用 |
| `dcp_local_seq_lens` | | decode context parallelism 本 rank 的序列长 |
| `logits_indices_padded` / `num_logits_indices` | | 结构化输出/快速 prefill |

派生方法：
- `batch_size()` = `seq_lens.shape[0]`；
- `naive_query_lens()` = `query_start_loc[1:] - query_start_loc[:-1]`；
- `compute_num_computed_tokens()` = `seq_lens - query_lens`（在 device 上算，避免 H↔D 同步）；
- `unpadded(num_actual_tokens, num_actual_reqs)`：CUDA graph 用 padding 张量，实际执行前切片到真实大小。

**关键设计**：尽可能保留 GPU 版本，CPU 副本只在必要时懒加载，且废弃同步属性（`seq_lens_cpu`、`num_computed_tokens_cpu` 被标记 deprecated），以兼容异步调度（async scheduling）。

> **query_len / context_len / seq_len 的关系**（[flash_attn.py:227-233](../../vllm/v1/attention/backends/flash_attn.py#L227-L233) 图示）：N-1 步及之前是 context_len；第 N 步新增的是 query_len；seq_len = context_len + query_len。prefill 时 query_len 可能很大（chunked prefill 的一个 chunk），decode 时 query_len = 1（或 1+spec tokens）。

---

## 四、AttentionMetadataBuilder：每层元数据构建

### 4.1 接口

[backend.py:516-663](../../vllm/v1/attention/backend.py#L516-L663)：

```python
class AttentionMetadataBuilder(ABC, Generic[M]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold: int | None = None
    supports_update_block_table: bool = False

    @abstractmethod
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device): ...

    @abstractmethod
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False) -> M: ...

    def update_block_table(self, metadata, blk_table, slot_mapping) -> M: ...
    def build_for_cudagraph_capture(self, common_attn_metadata) -> M: ...
    def build_for_drafting(self, common_attn_metadata, draft_index) -> M: ...
    def use_cascade_attention(self, common_prefix_len, query_lens, ...) -> bool: ...
```

- 每个 **KV cache group** 一个 builder 实例（构造时接收该 group 的所有 `layer_names`），同组层共享同一份 metadata；
- `build()` 把 `CommonAttentionMetadata` 转成后端专属的 `M`（如 FlashInfer 的 wrapper plan、FlashMLA 的 tile scheduler metadata）；
- `reorder_batch_threshold`：是否把 batch 重排为"decode 在前、prefill 在后"，以及阈值（query_len ≤ 阈值的请求被当作 decode 处理，让小 prefill 蹭 decode 快路径）。`_init_reorder_batch_threshold`（[backend.py:550-581](../../vllm/v1/attention/backend.py#L550-L581)）会根据 spec tokens 数和 DCP 调整；
- `fast_build=True`：草稿模型/投机解码中，为只用少数层/迭代的 metadata 优先构建速度而非执行速度；
- `supports_update_block_table` + `update_block_table`：多个 KV group 元数据几乎相同、仅 block table 不同时，复用已有 metadata 只换 block table，省重建开销。

### 4.2 batch 重排与 decode/prefill 拆分

`backends/utils.py` 提供一组被各 builder 复用的函数：

- `split_decodes_and_prefills`（[utils.py:507-556](../../vllm/v1/attention/backends/utils.py#L507-L556)）：按 query 长度把请求拆成 decode 段（前）和 prefill 段（后），返回索引和计数；
- `reorder_batch_to_split_decodes_and_prefills`（[utils.py:606-669](../../vllm/v1/attention/backends/utils.py#L606-L669)）：实际重排 query/输出张量；
- `split_prefill_chunks`、`make_local_attention_virtual_batches`：chunked local attention（Gemma3）等。

多数后端的 builder 遵循相同范式：重排 → 切片 seq_lens/query_start_loc → 构建 decode wrapper + prefill wrapper 的 plan → 组装成自己的 metadata dataclass。

---

## 五、AttentionImpl：执行前向

### 5.1 两个 Impl 基类

[backend.py:685-840](../../vllm/v1/attention/backend.py#L685-L840)：

- `AttentionImplBase`：公共属性（`num_heads/head_size/scale`、CP rank/world_size）和 `process_weights_after_loading`。`__new__` 中自动探测 DCP/PCP process group（[backend.py:731-757](../../vllm/v1/attention/backend.py#L731-L757)），计算 `need_to_return_lse_for_decode`（DCP 需要 softmax LSE 做跨 rank 合并）。
- `AttentionImpl(AttentionImplBase[T])`：标准注意力，抽象方法：
  - `__init__(num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window, kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name)`；
  - `forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)`。
  - 可选能力：`fused_output_quant_supported`、`fused_rope_kvcache_supported`、`do_rope_and_kv_cache_update`、`supports_quant_query_input`。

`AttentionLayer` Protocol（[backend.py:666-682](../../vllm/v1/attention/backend.py#L666-L682)）定义 impl 可见的层接口：`_q_scale/_k_scale/_v_scale`（量化缩放因子）、`_prob_scale`、`forward`。

### 5.2 MLA 的 Impl 基类

MLA（见第七节）有独立接口：
- `MLAAttentionImpl`（[backend.py:843-930](../../vllm/v1/attention/backend.py#L843-L930)）：稠密 MLA，抽象 `forward_mha`（prefill，MHA 风格）和 `forward_mqa`（decode，MQA 风格），并提供共享的 `do_kv_cache_update`（调用 `ops.concat_and_cache_mla`）；
- `SparseMLAAttentionImpl`（[backend.py:933-1010](../../vllm/v1/attention/backend.py#L933-L1010)）：稀疏 MLA，**只有 `forward_mqa`**（不支持 prefill）。

### 5.3 模型层如何调用 impl

模型的 `Attention.forward`（[model_executor/layers/attention/attention.py:409-501](../../vllm/model_executor/layers/attention/attention.py#L409-L501)）：
1. reshape q/k/v 为 3D；
2. 若后端 `forward_includes_kv_cache_update=False`，先调 `unified_kv_cache_update` 写 KV；
3. 调 `torch.ops.vllm.unified_attention_with_output`（或 direct call 路径）；
4. 该自定义 op 解析 forward context 中的 `attn_metadata[layer_name]`，调用 `self.impl.forward(self, query, key, value, kv_cache, attn_metadata, output=...)`（[attention.py:705-733](../../vllm/model_executor/layers/attention/attention.py#L705-L733)）。

`attn_metadata` 是一个 **`dict[layer_name, M]`**（多 KV group 时每层可能不同），由 model runner 在 `execute_model` 中通过 builder 构建并放入 forward context。`kv_cache_dummy_dep` 是一个空依赖张量，用数据依赖保证 `torch.compile` 下 KV 写入先于 attention 读取。

---

## 六、标准注意力后端

### 6.1 FlashAttentionBackend（参考实现）

[flash_attn.py](../../vllm/v1/attention/backends/flash_attn.py) 是最成熟、最易读的后端，建议作为阅读起点。

**Backend（[L69-224](../../vllm/v1/attention/backends/flash_attn.py#L69-L224)）**：
- KV cache 形状 `[2, num_blocks, block_size, num_kv_heads, head_size]`（K/V 合并在第 0 维）；
- 支持的 kernel block size 为 16 的倍数，`get_preferred_block_size` 有偏好逻辑；
- 支持 non-causal、encoder 类型、per-head quant scales、attention sink（取决于 FA 版本）、FP8 KV cache（需 `flash_attn_supports_fp8`）；
- Hopper 上用 FA3，支持 scheduler metadata（AOT 调度）。

**FlashAttentionMetadata（[L226-259](../../vllm/v1/attention/backends/flash_attn.py#L226-L259)）**：除公共字段外，还有 cascade 相关（`use_cascade`、`common_prefix_len`、`cu_prefix_query_lens`、`prefix_kv_lens`、`suffix_kv_lens`）、DCP（`dcp_context_kv_lens`）、AOT 调度（`scheduler_metadata`、`max_num_splits`）。

**Builder（[L279-595](../../vllm/v1/attention/backends/flash_attn.py#L279-L595)）**：
- CUDA graph 支持：有 scheduler_metadata（FA3/Hopper）时 `ALWAYS`，否则 `UNIFORM_BATCH`（[L298-302](../../vllm/v1/attention/backends/flash_attn.py#L298-L302)）；
- `supports_update_block_table = True`；
- `build()` 重排 batch、构建 cu_seqlens、调用 FA 的 scheduler metadata 构造、判断并构建 cascade 元数据。

**Impl.forward（[L677-896](../../vllm/v1/attention/backends/flash_attn.py#L677-L896)）**：
1. profiling 时直接填 0 返回；
2. encoder/encoder-only 类型走 `_forward_encoder_attention`（无 KV cache，双向）；
3. decoder/cross-attn：`kv_cache.unbind(0)` 拆 K/V，`canonicalize_singleton_dim_strides` 修正 size-1 维的退化 stride（TMA 16 字节对齐要求）；
4. 量化 KV cache 时 view 成 FP8 dtype；
5. 非 cascade 路径：调 `flash_attn_varlen_func`（prefill+decode 统一的 varlen 内核），传 `cu_seqlens_q`、`seqused_k`、`block_table`、`scheduler_metadata`、滑窗、softcap、alibi、k/v descale；
6. cascade 路径：调 `cascade_attn`（见 6.5）；
7. DCP 路径 `_forward_with_dcp`：all-gather LSE 后合并。

`do_kv_cache_update`（[L863-896](../../vllm/v1/attention/backends/flash_attn.py#L863-L896)）调用 `reshape_and_cache_flash` 把新 K/V 写入分页缓存（带 FP8 量化）。

### 6.2 FlashInferBackend

[flashinfer.py](../../vllm/v1/attention/backends/flashinfer.py) 是 Blackwell 上的高性能后端，最大特点是 **wrapper 对象 + plan/run 两阶段**，并集成 NVIDIA 的 TRT-LLM gen kernel。

**Backend（[L327-439](../../vllm/v1/attention/backends/flashinfer.py)）**：
- KV cache 形状与 FA 相同，但 `get_kv_cache_stride_order` 物理重排；
- `get_required_kv_cache_layout()` 在支持 TRT-LLM 时返回 `HND`；
- 支持 attention sink（需 TRT-LLM）；
- head size 白名单（受 FlashInfer 支持限制）。

**四类 metadata 子对象**（[L441-501](../../vllm/v1/attention/backends/flashinfer.py#L441-L501)）：`FIPrefill`/`FIDecode`（原生 FlashInfer wrapper）与 `TRTLLMPrefill`/`TRTLLMDecode`（TRT-LLM gen kernel），各自持有 plan 好的 wrapper 与 block_tables/seq_lens。`FlashInferMetadata`（[L503](../../vllm/v1/attention/backends/flashinfer.py#L503)）组合两者。

**Builder（[L540-1258](../../vllm/v1/attention/backends/flashinfer.py#L540-L1258)）**：
- 关键是 `_compute_flashinfer_kv_metadata`（[L829](../../vllm/v1/attention/backends/flashinfer.py#L829)）把 vLLM 的 block table 转成 FlashInfer 的 paged KV 格式，并对每个 wrapper 调 `plan()`；
- 延迟创建并缓存 wrapper（`_get_prefill_wrapper`、`_get_decode_wrapper`），用 `fast_plan_decode`（[L1857](../../vllm/v1/attention/backends/flashinfer.py#L1857)）在 CUDA graph 下只做 H→D 拷贝，避免 D→D；
- reorder 阈值初始为 1，支持 spec-as-decode 时按 spec tokens 放大。

**Impl.forward（[L1361-1825](../../vllm/v1/attention/backends/flashinfer.py#L1361-L1825)）**：decode token 排在前、prefill token 排在后，分别走 `decode_wrapper.run()` 和 `prefill_wrapper.run()`；TRT-LLM 路径用 `trtllm_batch_decode_with_kv_cache` / `trtllm_batch_context_with_kv_cache`，并处理 NVFP4（拆分 data/scale、FP8 输出缓冲再反量化）、FP8 Q/K/V scale、bmm1/bmm2 scale、attention sink、DCP（all-gather LSE）等。`do_kv_cache_update` 同样用 `reshape_and_cache_flash`。

### 6.3 TritonAttentionBackend（可移植后端）

[triton_attn.py](../../vllm/v1/attention/backends/triton_attn.py) 是纯 Triton 实现，作为无 FA/FlashInfer 时的回退，也支持更多特性组合。

- CUDA graph 支持 `ALWAYS`（[L126](../../vllm/v1/attention/backends/triton_attn.py#L126)）；
- 支持 mm_prefix（PrefixLM/多模态前缀双向）、sink、alibi_sqrt、encoder 类型、non-causal、batch invariance；
- forward 调 `ops/triton_unified_attention.py` 的统一 kernel（见第八节），它用一个 kernel 同时处理 prefill/decode，支持 2D/3D（分段并行 softmax）；
- `do_kv_cache_update` 调 Triton 版 `reshape_and_cache_flash`；
- 额外支持 `fused_rope_kvcache_supported` + `do_rope_and_kv_cache_update`（RoPE 与 KV 写入融合）。

### 6.4 其他标准后端

| 后端 | 文件 | 特点 |
|------|------|------|
| `FlashAttentionDiffKVBackend` | flash_attn_diffkv.py | K/V 的 head_dim 不同，写入合并的 KV 张量（`[..., head_k + head_v]`） |
| `FlexAttentionBackend` | flex_attention.py | 基于 PyTorch `flex_attention`，用 `BlockMask` 表达任意 mask（causal/bidirectional/sliding window/prefix-LM）。适合研究/自定义，性能不如专用 kernel。CG 支持 ALWAYS |
| `CPUAttentionBackend` | cpu_attn.py | 唯一支持全部 4 种 attention 类型的 CPU 后端；block size 为 16 倍数；head_size 32–512 |
| `RocmAttentionBackend` / `RocmAiter*` | rocm_*.py | AMD ROCm，AITER 库 |
| `TurboQuantAttentionBackend` | turboquant_attn.py | **标准 softmax attention 但 KV cache 被压缩**（3/4-bit）。prefill 在未压缩 K/V 上算后量化存储，小 continuation 可直接走 decode kernel。`forward_includes_kv_cache_update=False` |

### 6.5 Cascade Attention（级联注意力）

Cascade attention（[论文](https://arxiv.org/abs/2501.01005)）利用所有 running 请求的**公共前缀**：当一批 decode 请求共享一段长前缀时，前缀部分用一个 prefill kernel 一次性算好，后缀（各自不同部分）用 decode kernel 算，再用 LSE 加权合并。

- Builder 的 `use_cascade_attention()`（如 [flash_attn.py:1067](../../vllm/v1/attention/backends/flash_attn.py#L1067)）根据 `common_prefix_len`、query_lens、SM 数等判断是否值得；
- `cascade_attention()`（[flash_attn.py:1145](../../vllm/v1/attention/backends/flash_attn.py#L1145)）分别算 prefix/suffix，调用 `merge_attn_states` 合并；
- 公共前缀长度来自调度器输出的 `num_common_prefix_blocks`（见 [KV Cache 章节](./02_scheduler_kv_cache.md)），由 `KVCacheManager.get_num_common_prefix_blocks` 计算；
- 合并 kernel 有 C++ 版（`_custom_ops.merge_attn_states`）和 Triton 版（`ops/triton_merge_attn_states.py`），按数值稳定的在线 softmax 方式合并：`out = p_out*exp(p_lse-max) + s_out*exp(s_lse-max)`。

---

## 七、MLA 后端（DeepSeek V2/V3/V4）

MLA（Multi-head Latent Attention）是 DeepSeek 系列的核心优化。理解它需要先理解一个关键事实：

> **MLA 的公共逻辑大部分不在 `vllm/v1/attention/backends/mla/`，而在 [model_executor/layers/attention/mla_attention.py](../../vllm/model_executor/layers/attention/mla_attention.py)（2300+ 行）。** `mla/` 目录下只有各 kernel 的薄后端。

### 7.1 MLA 原理（简述）

MLA 每个 token 只缓存一个**低秩隐向量**而非每头 K/V：
- DeepSeek V3：`kv_lora_rank=512`、`qk_nope_head_dim=128`、`qk_rope_head_dim=64`、`v_head_dim=128`；
- 缓存表示为 `[kv_c_normed (512), k_pe (64)]` 拼接，`head_size = 576`，`num_kv_heads = 1`（MQA 形状）。

两条计算路径（详见 mla_attention.py 顶部 188 行 docstring）：
- **prefill（MHA，`forward_mha`）**：用 `kv_b_proj`（= `[W_UK; W_UV]`）把隐向量上投影为完整 `k_nope`/`v`，广播拼接 `k_pe`，跑标准 MHA（QK dim 192，V dim 128）；
- **decode（MQA，`forward_mqa`）**：把 `W_UK` **吸收进 query**（`ql_nope = q_nope @ W_UK_T`），直接对压缩缓存做 MQA（QK dim 576，V dim 512），attention 后再用 `W_UV` 上投影输出。这避免了从 HBM 搬运 N 头的 K/V。

chunked prefill 时整个 context 上投影会 OOM，因此在固定大小的 workspace 中分块处理，用 `merge_attn_states` 合并。

### 7.2 类层次

```
AttentionBackend (backend.py)
└── MLACommonBackend (mla_attention.py:1180)        # get_kv_cache_shape=[num_blocks, block_size, head_size]
    ├── FlashAttnMLABackend / FlashMLABackend / FlashInferMLABackend
    ├── TritonMLABackend / CutlassMLABackend / TokenspeedMLABackend
    └── AiterMLABackend (ROCm) ...

AttentionImplBase
├── MLAAttentionImpl (backend.py:843)               # 抽象 forward_mha + forward_mqa + do_kv_cache_update
│   └── MLACommonImpl (mla_attention.py:1939)       # forward_mha 公共实现 + chunked context
│       └── 各 kernel 后端只实现 forward_mqa
└── SparseMLAAttentionImpl (backend.py:933)         # 只有 forward_mqa
    └── FlashMLASparseImpl / FlashInferMLASparseImpl / ROCMAiterMLASparseImpl ...
```

**关键设计：吸收/上投影逻辑在公共层 `MLAAttention.forward_impl`（[mla_attention.py:594-824](../../vllm/model_executor/layers/attention/mla_attention.py#L594-L824)），各后端 kernel 只做 `forward_mqa`。** 权重在 `process_weights_after_loading` 中预处理为 `W_UK_T (N,P,L)` 和 `W_UV (N,L,V)`。

### 7.3 KV cache 形状与特殊 FP8 布局

- 标准稠密：`(num_blocks, block_size, head_size)`，3D 无 num_kv_heads 维（MLA 单 KV 头）；
- DeepSeek 自定义 FP8（`fp8_ds_mla`）每 token 不是 576 字节：
  - **V3.2：656 字节/token** — 512 字节 fp8 NoPE + 16 字节 fp32 block scale（每 128 元素一个，共 4）+ 128 字节 bf16 RoPE；
  - **V4：584 字节/token** — 448 字节 fp8 NoPE + 128 字节 bf16 RoPE + 8 字节（7 个 ue8m0 scale + 1 pad）。
- 这些布局体现在 `FlashMLASparseBackend.get_kv_cache_shape`（返回 656/584）和 `MLAAttentionSpec.real_page_size_bytes` 中。

`do_kv_cache_update` 调 `ops.concat_and_cache_mla(kv_c_normed, k_pe, kv_cache, slot_mapping, ...)` 把两部分连续写入缓存。

### 7.4 prefill 子注册表

MLA 的 prefill 和 decode 用**不同 kernel 库**，因此 vLLM 把 prefill 后端独立成子系统（`mla/prefill/`）：

- `MLAPrefillBackend` 基类（[prefill/base.py](../../vllm/v1/attention/backends/mla/prefill/base.py)）：抽象 `run_prefill_new_tokens`（causal）和 `run_prefill_context_chunk`（non-causal，返回 LSE 用于合并）；
- `MLAPrefillBackendEnum`（prefill/registry.py）：`FLASH_ATTN`、`FLASHINFER`、`TRTLLM_RAGGED`、`TOKENSPEED_MLA`；
- `prefill/selector.py` 按 GPU 代际选：**所有 GPU 默认 FlashAttention prefill**；Blackwell 上 TRT-LLM Ragged / FlashInfer / TokenSpeed 作为备选；
- 实现：`prefill/flash_attn.py`（处理 QK dim 192 ≠ V dim 128，必要时 zero-pad）、`prefill/flashinfer.py`（Blackwell，R1 维度，最多 30 个 chunk wrapper）、`prefill/trtllm_ragged.py`、`prefill/tokenspeed_mla.py`。

`MLACommonMetadataBuilder.build`（mla_attention.py:1589-1863）把 batch 拆成 decode/prefill，为有 context 的 prefill 计算 chunking（workspace 大小上限 64K token），构造 `MLACommonPrefillMetadata` 并调 `prefill_backend.prepare_metadata`。

### 7.5 稠密 MLA 各后端对比

| Backend | SM | decode kernel | 特点 |
|---------|----|---------------|------|
| `FlashAttnMLA` | 9 Hopper | `flash_attn_varlen_func`（FA3） | 参考实现，MQA 技巧：K=RoPE 部分、V=隐向量、吸收后的 q_nope 作为 `q_v`；不支持 FP8 KV |
| `FlashMLA` | 9/10 | `flash_mla_with_kvcache` | DeepSeek 官方 FlashMLA，tile-scheduler metadata；Hopper 主力 |
| `FlashInferMLA` | 10 Blackwell | `trtllm_batch_decode_with_kv_cache_mla` | 默认，需 HND layout，支持 FP8/NVFP4 |
| `CUTLASSMLA` | 10 | `sm100_cutlass_mla_decode` | 强制 block 128，head pad 到 128，SM100 workspace |
| `TritonMLA` | 任意 | `decode_attention_fwd(is_mla=True)` | 可移植回退，自适应 split-K，BF16 Q 内核反量化 KV |
| `TokenspeedMLA` | 10 | 第三方 CuTe DSL | FP8 only，R1 维度 |
| `AiterMLA` (ROCm) | — | AMD `aiter.mla_decode_fwd` | 内部 flatten 到 page_size=1 |

### 7.6 稀疏 MLA（DeepSeek V3.2 / V4）

稀疏 MLA 在 decode 时**不看全部 KV**，而是由一个独立的轻量 **indexer** 给所有候选位置打分、选 top-k（如 `index_topk=2048`），主 attention 只 gather 这些行。V4 还结合：
- SWA（滑动窗口）处理近期 token；
- 压缩注意力：C4A（4x 压缩）、C128A（128x 压缩）KV cache 处理远端 token（`compress_ratios` 配置）。

组件：
- **indexer**（`mla/indexer.py`）：`DeepseekV32IndexerBackend` 是个特殊的 `AttentionBackend` 但**无 Impl**——真正的打分在模型 `Indexer` 模块 + `sparse_attn_indexer` 自定义 op 中，结果写入共享的 `topk_indices_buffer`。metadata builder 处理 MTP/spec 展开、V4 压缩序列长、prefill chunking；
- **FlashMLASparseImpl**（`flashmla_sparse.py`）：主稀疏 decode 后端，构造时持有 `indexer.topk_indices_buffer`；`forward_mqa` 读取 top-k 索引，用 `triton_convert_req_index_to_global_index`（sparse_utils.py）把每请求逻辑位置转成全局分页 slot，再调 `flash_mla_sparse_fwd` / `flash_mla_with_kvcache`。FP8 kernel 把 cache 当 uint8 解析 656/584 字节打包格式；
- **FlashInferMLASparse**（Blackwell）：用标准 FP8（非 `fp8_ds_mla`），调 TRT-LLM MLA kernel 传稀疏索引；
- **sparse_swa.py**：V4 SWA KV cache 的辅助后端，用 Triton kernel 计算每 decode token 的窗口索引，并按层类型（swaonly/c4a/c128a，由 compress_ratio 映射）构建共享 tile-scheduler plan；
- **compressor_utils.py / sparse_utils.py**：压缩 slot 映射、逻辑→全局索引转换。

稀疏后端都继承 `SparseMLAAttentionImpl`（**只有 `forward_mqa`，无 prefill**）；稀疏模型的 prefill 由 BF16 prefill kernel 从 FP8 cache 反量化，或用混合 batch FP8 decode kernel 处理。

---

## 八、低层算子 `ops/`

后端 impl 调用的 Triton/C++ 算子，分三类：

### 8.1 KV 写入

- **`triton_reshape_and_cache_flash.py`**：
  - 标准 Flash 布局 kernel：5D K `[block, head, dim//x, slot, x]` + 4D V `[block, head, dim, slot]`，按 `slot_mapping` 写入，支持 FP8 量化；
  - DiffKV 变体：K/V head_dim 不同，写合并张量；
  - **per-token-head 动态量化**变体：每 (token, head) 独立算 absmax scale，存独立 scale 张量（int8/FP8）；
- **`paged_attn.py`**：`PagedAttention.write_to_paged_cache` 封装 C++ `reshape_and_cache`，并提供 `split_kv_cache` 把合并 KV 视图拆成 FlashAttention 风格的 5D/4D。

### 8.2 注意力计算

| 文件 | 作用 |
|------|------|
| `triton_unified_attention.py` | 最新统一 kernel：一个 kernel 处理 prefill+decode，2D（直接输出）或 3D（长序列分段并行 softmax，再 `reduce_segments` 归约）。支持 causal/sliding window/chunked-local/mm_prefix/sink/ALiBi/softcap/FP8 |
| `triton_decode_attention.py` | split-KV 两阶段 decode：stage1 各 SM 算部分结果+LSE，stage2 跨 split 归约。有 MHA 和 GQA/MQA/MLA grouped 版本 |
| `triton_prefill_attention.py` | 无分页（page=1）的 varlen prefill，支持双向 sliding window（ViT 等） |
| `prefix_prefill.py` | **cascade attention 核心**：prefix（已缓存 KV，非因果）+ suffix（query 自身，因果）两段循环，在线 softmax 自然合并，无需额外 merge kernel；有 ALiBi 变体 |
| `chunked_prefill_paged_decode.py` | ROCm 为主的统一 chunked-prefill + paged-decode，非 2 幂 block size（如 544）用 `PHYSICAL_BLOCK_SIZE`/`BLOCK_SIZE` 分离 |
| `flashmla.py` | FlashMLA C++ 扩展绑定（`flash_mla_with_kvcache`、`get_mla_metadata`、FP8 扩展） |

### 8.3 合并与通信

- `merge_attn_states.py` / `triton_merge_attn_states.py`：prefix+suffix 的 LSE 加权合并（cascade）；优先 C++ kernel，回退 Triton；
- `common.py`：**Context Parallelism** 后处理——all-gather 各 rank 的 LSE，用 `exp(local_lse - global_lse)` 修正本地输出；提供 AllGather+ReduceScatter / AllGather+AllReduce 变体，以及序列 pack/unpack；
- `dcp_alltoall.py`：**Decode Context Parallelism** 的 all-to-all 方案，把 (output, LSE) 打包单次 All-to-All 交换，接收端按 LSE 加权合并（替代 AG+RS）；
- `deepseek_v4_ops/`：DeepSeek V4 稀疏 MLA 的融合算子——`quantize_and_insert_k_cache`/`dequantize_and_gather_k_cache`、indexer Q 的融合 RoPE+量化（MXFP4）、compressor 的 RMSNorm+RoPE+量化+缓存插入、Q/KV RMSNorm 融合等；
- `vit_attn_wrappers.py`：ViT 注意力（无 KV cache、双向）的 `torch.compile` 兼容包装（FA/Triton/SDPA/FlashInfer cuDNN）。

---

## 九、SSM / 线性注意力后端

`mamba_attn.py`、`mamba1_attn.py`、`mamba2_attn.py`、`linear_attn.py`、`gdn_attn.py`、`short_conv_attn.py` 这些后端**继承 `AttentionBackend` 接口但不做 softmax attention**——它们是状态空间模型（SSM）或线性注意力，复用 v1 的 metadata/build/KV-cache 基础设施，但 `is_ssm()` 返回 True。

它们用 `MambaSpec`（而非 `AttentionSpec`）描述缓存：缓存的不是 K/V 张量，而是 **RNN/SSD 隐状态**。

### 9.1 Mamba cache 模式

`mamba_cache_mode`（config）：
- **`"all"`**：为整个 max_model_len 预分配 block table（`state_indices_tensor_d` 形状 `[max_bs, max_num_blocks]`），支持 prefix caching，追踪 last computed/scheduled block index；
- **`"none"`/其他**：只存当前/最近几个状态槽（形状 `[max_bs, 1+num_spec_tokens]`）。

prefill（`state_indices_tensor_p`，一维，每序列一个起始状态）与 decode（`state_indices_tensor_d`，二维，spec decode 每步一个槽）的状态索引分离。

### 9.2 公共基类

[backends/mamba_attn.py](../../vllm/v1/attention/backends/mamba_attn.py) 提供：
- `BaseMambaAttentionMetadata`：计数、prefill/decode 状态索引、prefix caching block 索引、Mamba2 SSD 的 chunk metadata、causal_conv1d Triton 元数据；
- `BaseMambaAttentionMetadataBuilder`（CG 支持 `UNIFORM_BATCH`）：`split_decodes_and_prefills` 分类、获取 state indices、处理 prefix caching block index、为 prefill 计算 causal_conv1d metadata；decode 张量拷入预分配静态 buffer 以支持 CUDA graph；
- chunk metadata 保证每个 chunk 只含单序列 token、且每 chunk_size token 能取一次 mamba 状态。

### 9.3 各变体

| 后端 | 文件 | 特点 |
|------|------|------|
| `Mamba1` | mamba1_attn.py | 极简，`all` 模式下额外构建 `cu_chunk_seqlen_p`/`last_chunk_indices_p` |
| `Mamba2` | mamba2_attn.py | SSD，按序列边界和物理 chunk 边界双重切分，必须设 `mamba_chunk_size`，有 initial states 时 `prep_initial_states=True` |
| `ShortConv` | short_conv_attn.py | 复用 Mamba base（causal_conv1d 无 chunk 概念） |
| `Linear` | linear_attn.py | 单矩阵状态，CG `UNIFORM_SINGLE_TOKEN_DECODE`，每序列单个状态索引，无 chunk/prefix 追踪 |
| `GDN` (GatedDeltaNet) | gdn_attn.py | 用 FLA chunk ops；区分 spec/non-spec decode 并用 argsort 重排；FLA chunk metadata；两者并存时把 non-spec decode 重归类为 prefill |

这些后端的实际 SSM 计算在模型层（`vllm/model_executor/layers/mamba/`），后端主要负责把分页状态索引喂给 SSM op。

---

## 十、关键设计决策总结

1. **三层对象分工**：Backend（静态能力+工厂）、Builder（每 group 一份，构建元数据）、Impl（每层一份，执行）。能力声明集中在 Backend，使自动选择只需检查类方法而无需实例化。

2. **平台优先级 + 缓存选择**：后端不是硬编码，而是平台对象持有按硬件代际/模型特性排序的候选列表，`validate_configuration` 裁判，结果按 selector config 缓存。第三方可用 `register_backend` 或 `CUSTOM` 注入。

3. **CommonAttentionMetadata 只构建一次**：跨层共享的 query_start_loc/seq_lens/block_table/slot_mapping 由 model runner 统一构建；Builder 只做每层/每 group 特有的轻量转换，并尽量在 GPU 上、避免 H↔D 同步以支持异步调度。

4. **KV cache 形状由后端决定**：`get_kv_cache_shape` + `get_kv_cache_stride_order` 让不同 kernel 用最优布局（NHD/HND、合并 KV、MLA 3D、656/584 字节 FP8 打包），model runner 据此分配张量。

5. **统一 prefill/decode**：标准后端用 varlen kernel（FA/FlashInfer/Triton unified）在一次 forward 内同时处理 prefill 和 decode，靠 `query_start_loc` 区分；batch 重排让 decode 在前、prefill 在后。

6. **MLA 的 prefill/decode 解耦**：decode 走 MQA + W_UK 吸收，prefill 走 MHA + kv_b_proj 上投影，二者可由不同 kernel 库实现（独立子注册表）。公共的吸收/上投影/chunked context 逻辑在 `MLACommonImpl`，各 kernel 只实现 `forward_mqa`。

7. **级联注意力利用公共前缀**：调度器提供 `num_common_prefix_blocks`，后端判断收益后把 prefix/suffix 分开算再用 LSE 合并，显著降低长共享前缀 decode 的计算量。

8. **分页寻址两种布局**：FlashAttention 风格 5D K + 4D V（向量化加载）与通用 4D 合并布局；所有 Triton kernel 用 stride 参数兼容。非 2 幂 block size（如 544）通过 `BLOCK_SIZE`（tile）/`PHYSICAL_BLOCK_SIZE`（cache 块）分离支持。

9. **CUDA graph 分级与 metadata 复用**：ALWAYS / UNIFORM_BATCH / UNIFORM_SINGLE_TOKEN_DECODE / NEVER 四级；`update_block_table` 允许多 KV group 复用近同元数据；cudagraph capture 用 padding + 静态 buffer。

10. **SSM 复用注意力基础设施**：Mamba/线性注意力实现同一个 `AttentionBackend` 接口（`is_ssm()=True`），用 `MambaSpec` 缓存隐状态而非 K/V，使引擎主循环无需区分"注意力"和"状态空间模型"。

---

## 十一、与其他模块的关系

| 模块 | 交互方式 |
|------|----------|
| `vllm/v1/core/`（调度器） | 调度器输出的 `num_scheduled_tokens`、`block_table`、`slot_mapping`、`num_common_prefix_blocks` 是 CommonAttentionMetadata 的来源 |
| `vllm/v1/worker/gpu_model_runner.py` | 构建 CommonAttentionMetadata、按 KV group 调 builder.build()、按 `get_kv_cache_shape` 分配 KV cache、把 metadata 放入 forward context |
| `vllm/model_executor/layers/attention/` | `Attention`/`MLAAttention` 等模型层持有 impl，在 forward 中调 `impl.forward`；通过 `get_attn_backend` 选后端 |
| `vllm/v1/kv_cache_interface.py` | `KVCacheSpec`/`AttentionSpec`/`MLAAttentionSpec` 等描述层的 KV 需求，决定 builder 构造参数 |
| `vllm/platforms/` | 各平台提供 `get_attn_backend_cls` 候选优先级与硬件能力探测 |
| `vllm/_custom_ops` / `csrc/` | FlashAttention、reshape_and_cache、merge_attn_states、concat_and_cache_mla 等 C++/CUDA 算子 |
| `vllm/v1/attention/ops/` | Triton 算子（unified/decode/prefill attention、CP 通信、DeepSeek V4 融合） |
| `vllm/model_executor/layers/quantization/` | KV cache 量化（FP8/NVFP4/per-head scales）与后端 `supports_*` 能力协商 |
