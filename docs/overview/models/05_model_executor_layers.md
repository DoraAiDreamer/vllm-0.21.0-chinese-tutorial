# 模型执行与层

> 源码路径: `vllm/model_executor/layers/`

上一篇 Worker/ModelRunner 解决的是"一个 step 怎么在 GPU 上跑起来"。本篇进入模型**内部**：`self.model(...)` 里那一个个 `nn.Module`——线性层、Attention、归一化、激活、RoPE、Embedding、MoE、Mamba/SSM、池化头等。它们由各模型文件（`model_executor/models/`）拼装，但真正干活的"积木"都定义在这里。

vLLM 层设计有三条贯穿始终的主线：

1. **张量并行（TP）内置在层里**：`ColumnParallelLinear`/`RowParallelLinear`/`VocabParallelEmbedding` 在构造时就按 rank 切权重、在 forward 里插入 all-gather/all-reduce，对模型代码透明；
2. **量化是可插拔的 "method"**：每个 Linear/MoE/Attention 持有一个 `quant_method`，它负责 `create_weights`（建什么样的参数张量）和 `apply`（forward 怎么算），未量化时就是朴素 GEMM；
3. **自定义算子体系**：`CustomOp`（按设备分派 `forward_native/cuda/hip/cpu/...`）和 `PluggableLayer`（可被 OOT 后端整体替换），让同一层在 CUDA/ROCm/CPU/XPU/TPU 上各走最优 kernel，同时能被 `torch.compile` 当作不透明算子处理。

> 注意力的"后端选择/metadata 构建/impl 执行"已在 [03 注意力后端](../core/03_attention_backends.md) 讲过；量化的完整后端（FP8/GPTQ/AWQ/GGUF/Marlin…）是第 11 篇。本篇聚焦层本身的结构、TP 切分、权重加载和 forward 数据流。

---

## 整体结构

```
vllm/model_executor/layers/
├── linear.py                    # LinearBase + Replicated/ColumnParallel/RowParallel/Merged/QKVParallel
├── vocab_parallel_embedding.py  # VocabParallelEmbedding + ParallelLMHead（TP 切词表）
├── layernorm.py                 # RMSNorm / GemmaRMSNorm / RMSNormGated / LayerNorm / PolyNorm
├── activation.py                # SiluAndMul / GeluAndMul / GELU / NewGELU / FatreluAndMul / get_act_fn ...
├── conv.py                      # Conv2dLayer 等（视觉模型用）
├── rotary_embedding/            # RoPE 家族（base + Llama3/YaRN/DynamicNTK/MRoPE/XDRoPE/Gemma4/...）
├── attention/
│   ├── attention.py             # Attention 层（持有 backend impl + kv_cache，forward 调 unified_attention）
│   ├── mla_attention.py         # MLAAttention（DeepSeek V2/V3/V4 的低秩注意力）
│   ├── cross_attention.py       # encoder-decoder 交叉注意力
│   ├── chunked_local_attention.py / static_sink_attention.py
│   ├── encoder_only_attention.py / mm_encoder_attention.py
│   └── kv_transfer_utils.py
├── mla.py                       # MultiHeadLatentAttentionWrapper：MLA 的投影/解压/RoPE 外层
├── deepseek_v4_attention.py     # DeepSeek V4 稀疏 MLA 专用层
├── lightning_attn.py            # Lightning Attention（线性注意力变体）
├── kda.py                       # Kimi Delta Attention（线性注意力/SSM 混合）
├── mhc.py                       # mHC 融合块（MoE hybrid compute，深度融合 GEMM）
├── deepseek_compressor.py       # KV cache 压缩后端（CompressorBackend）
├── sparse_attn_indexer.py       # 稀疏注意力索引器
├── resampler.py                 # 多模态 perceiver resampler（Flamingo 风格）
├── fused_moe/                   # MoE：FusedMoE 层 + 路由器 + 各硬件 fused kernel
│   ├── layer.py                 # FusedMoE(nn.Module)：权重/路由/EPLB/quant_method
│   ├── fused_moe.py             # 经典 Triton fused MoE kernel 与分发
│   ├── fused_batched_moe.py / triton_cutlass_moe.py / triton_deep_gemm_moe.py
│   ├── experts/                 # 各后端专家计算（cutlass/deepgemm/flashinfer/marlin/triton/rocm/cpu/xpu...）
│   ├── router/                  # top-k 路由、grouped topk、score 修正
│   ├── runner/                  # MoERunner：把路由+permute+GEMM+reduce 串起来；SharedExperts
│   ├── prepare_finalize/        # token 排列/反排列（all-to-all 前后）
│   ├── all2all_utils.py         # EP 的 all-to-all 通信
│   ├── config.py / moe_align_block_size.py / utils.py ...
│   └── lora_context.py / lora_experts_mixin.py / routed_experts_capturer.py
├── mamba/                       # Mamba / 状态空间模型
│   ├── abstract.py              # MambaBase(AttentionLayerBase)
│   ├── mamba_mixer.py           # MambaMixer（Mamba-1 SSM）
│   ├── mamba_mixer2.py          # MambaMixer2（Mamba-2，SSD）
│   ├── gdn_linear_attn.py       # Gated DeltaNet 线性注意力
│   ├── linear_attn.py / short_conv.py
│   ├── lamport_workspace.py / mamba_utils.py
│   └── ops/                     # causal_conv1d / ssd_* / ssu_dispatch / layernorm_gated（Triton/C++/CPU）
├── pooler/                      # 池化/分类/奖励模型头
│   ├── abstract.py              # Pooler 基类
│   ├── activations.py           # PoolerClassification/normalize 等
│   ├── seqwise/                 # 序列级池化：EmbeddingPoolerHead / ClassifierPoolerHead
│   ├── tokrise/                 # token 级池化
│   ├── special.py               # BgeM3 / BOSEOS / Dispatch / Identity pooler
│   └── common.py
├── fla/                         # "Flash Linear Attention" 算子（第三方 fla 库的集成）
├── quantization/                # 量化 method 实现（第 11 篇详述）
├── batch_invariant.py           # 与 batch 形状无关的持久化 Triton kernel（GEMM/RMSNorm/softmax/...）
├── logits_processor.py          # 模型内置 logits processor 基类
└── utils.py                     # GEMM 分发、vLLMParameter 工具等
```

---

## 一、Linear 层与量化 method

所有线性层的核心是 [linear.py](../../vllm/model_executor/layers/linear.py)。它用"组合优于继承"的方式把量化解耦：

```
nn.Module
   └── PluggableLayer
         └── LinearBase(quant_method: QuantizeMethodBase)
               ├── ReplicatedLinear          # 不切分，TP 各 rank 都有一份
               ├── ColumnParallelLinear      # 按输出维切（A=[A_1..A_p]），可选 all-gather
               │     ├── MergedColumnParallelLinear  # 把 gate/up 等多个矩阵打包
               │     └── QKVParallelLinear         # 把 Q/K/V 打包并分别按 head 切
               └── RowParallelLinear         # 按输入维切，结果 all-reduce
```

### 1.1 LinearMethodBase：量化的两个钩子

[linear.py:142-180](../../vllm/model_executor/layers/linear.py#L142-L180)：

```python
class LinearMethodBase(QuantizeMethodBase):
    @abstractmethod
    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs): ...
    @abstractmethod
    def apply(self, layer, x, bias=None) -> torch.Tensor: ...
```

- `create_weights`：在 layer 上注册参数。未量化时（`UnquantizedLinearMethod`，[L186](../../vllm/model_executor/layers/linear.py#L186)）就是一个 `[sum(output_partition_sizes), input_size_per_partition]` 的 `ModelWeightParameter`；量化方法可注册成任意打包张量（如 Marlin 的打包权重、FP8 的 weight+scale）。
- `apply`：forward 时真正算 GEMM。未量化走 `dispatch_unquantized_gemm()(layer, x, weight, bias)`（按平台/形状选 cuBLAS/CUTLASS/Triton），开启 `VLLM_BATCH_INVARIANT` 时走 [batch_invariant.py](../../vllm/model_executor/layers/batch_invariant.py) 的持久化 Triton kernel。

每个层的 `weight_loader` 也作为 `extra_weight_attrs` 传给 `create_weights`，从而不同量化格式可以注册自己的加载逻辑。

### 1.2 ColumnParallelLinear

[linear.py:414](../../vllm/model_executor/layers/linear.py#L414)：把权重按**输出维**（dim=0）切成 `tp_size` 份，每 rank 持有 `[output_size/tp, input_size]`。

- forward（[L582-600](../../vllm/model_executor/layers/linear.py#L582-L600)）：`Y_i = X A_i (+b_i)`；若 `gather_output=True` 再 `tensor_model_parallel_all_gather` 拼回完整 `Y`，否则直接返回分片（典型用于 attention 的 QKV、FFN 的 gate/up，后续本就要按 head 切分）。
- weight_loader（[L537-572](../../vllm/model_executor/layers/linear.py#L537-L572)）：加载完整权重时，按 `param.output_dim` 用 `narrow` 切出本 rank 的 shard 再 copy；GGUF 还会在 `UninitializedParameter.materialize` 时就按 tp 缩小形状。Marlin/block-scale 等走 `weight_loader_v2`（`param.load_column_parallel_weight`）。

### 1.3 RowParallelLinear

[linear.py:1396](../../vllm/model_executor/layers/linear.py#L1396)：把权重按**输入维**（dim=1）切，每 rank 持有 `[output_size, input_size/tp]`。

- forward（[L1544-1570](../../vllm/model_executor/layers/linear.py#L1544-L1570)）：若 `input_is_parallel=False` 先把输入沿 last dim 切开；各 rank 算 `X_i A_i`；若 `reduce_results=True` 做 all-reduce 求和得到完整 `Y`。bias 只在 rank 0 加（避免重复加）。这是 FFN down_proj / attention o_proj 的典型形态——与上游 ColumnParallel 配对，一次 all-gather + 一次 all-reduce 构成一个 TP 组。
- 注意：`not reduce_results` 时不允许 bias（会导致结果不正确，[L1481](../../vllm/model_executor/layers/linear.py#L1481)）。

### 1.4 Merged 与 QKV 打包

- **`MergedColumnParallelLinear`**（[L611](../../vllm/model_executor/layers/linear.py#L611)）：把若干逻辑矩阵（如 SwiGLU 的 gate_proj 和 up_proj）在输出维拼接成一个大权重，一次 GEMM 出结果再 split。`output_sizes` 记录各逻辑分片宽度，`weight_loader` 根据 checkpoint 的 `shard_id`（"gate"/"up" 或索引）把权重写到拼接张量的正确偏移；还支持从 checkpoint 直接加载融合好的 w13（`_load_fused_module_from_checkpoint`）。
- **`QKVParallelParallelLinear`**（[L979](../../vllm/model_executor/layers/linear.py#L979)）：类似但处理 Q/K/V 三个头数不同的矩阵。Q 按 `num_heads` 切、K/V 按 `num_kv_heads` 切（GQA/MQA），`shard_id` 为 "q"/"k"/"v"，加载时分别计算在打包张量里的 offset/size。

`output_partition_sizes` 是贯穿三者的关键概念：本 rank 上每个逻辑矩阵的输出宽度列表，量化方法据此创建正确形状的权重。

### 1.5 vLLMParameter 与权重加载体系

线性层权重用的不是普通 `nn.Parameter`，而是 `ModelWeightParameter`/`BasevLLMParameter`（位于 `vllm/model_executor/parameter.py`）。它们在 `set_weight_attrs` 里携带 `input_dim/output_dim`、`weight_loader`、`is_sharded_weight` 等元信息。模型实现里每个层都定义 `weight_loader`，由全局的 `model.load_weights(...)` 按 state_dict 名调用，完成 TP 切分、打包、量化 scale 加载、GGUF 物化等。`WEIGHT_LOADER_V2_SUPPORTED` 列出的量化方法走更快的 v2 路径（直接 `param.load_*_weight`）。

---

## 二、归一化与激活

### 2.1 RMSNorm 家族

[layernorm.py](../../vllm/model_executor/layers/layernorm.py) 全部继承 `CustomOp`，按设备分派：

- **`RMSNorm`**（[L38](../../vllm/model_executor/layers/layernorm.py#L38)）：`y = w * x / sqrt(mean(x²) + eps)`。`forward_native` 走 `ir.ops.rms_norm`（中间表示算子，可被编译/降级），`forward_cuda` 在 `VLLM_BATCH_INVARIANT` 下走持久化 Triton kernel；支持 `fused_add_rms_norm`（传入 residual 时返回 `(normed, residual)`，把残差加和归一化融合）。
- **`GemmaRMSNorm`**（[L133](../../vllm/model_executor/layers/layernorm.py#L133)）：Gemma 用 `x*(1+w)` 且 dtype 处理不同。
- **`RMSNormGated`**（[L183](../../vllm/model_executor/layers/layernorm.py#L183)）：带可选 SiLU/Sigmoid 门控的 RMSNorm（Jamba 等混合模型、Mamba-2 用），支持 group norm；CUDA 走 `fla` 算子。
- **`LayerNorm`**（[L304](../../vllm/model_executor/layers/layernorm.py#L304)）：标准 LayerNorm（始终 float32 计算）。
- **`poly_norm`**：Cohere 等用的多项式归一化，走 custom op。

### 2.2 激活函数

[activation.py](../../vllm/model_executor/layers/activation.py) 里绝大多数是 `*AndMul` 形态——把"激活"和"逐元素乘"融合成一个 kernel（典型用于 SwiGLU：`silu(gate) * up`）：

| 类 | 作用 |
|----|------|
| `SiluAndMul`（[L118](../../vllm/model_executor/layers/activation.py#L118)） | `silu(x[..., :half]) * x[..., half:]`，LLaMA/Mistral 等最常用 |
| `GeluAndMul`（[L322](../../vllm/model_executor/layers/activation.py#L322)） | GELU 版本 |
| `SiluAndMulWithClamp`（[L155](../../vllm/model_executor/layers/activation.py#L155)） | DeepSeek V3 风格，带输出 clamp |
| `GeluAndMulSparse`（[L235](../../vllm/model_executor/layers/activation.py#L235)） | 带激活稀疏化（ReluFusion） |
| `GELU/NewGELU/FastGELU/QuickGELU` | 各种 GELU 近似 |
| `FatreluAndMul`（[L79](../../vllm/model_executor/layers/activation.py#L79)） | 带阈值的 ReLU |
| `ReLUSquaredActivation`、`XIELU`、`SwigluOAIAndMul`、`ScaledActivation` | 其他模型专用激活 |

工厂函数 `get_act_fn(name)`（[L736](../../vllm/model_executor/layers/activation.py#L736)）和 `get_act_and_mul_fn(name)`（[L764](../../vllm/model_executor/layers/activation.py#L764)）按配置字符串返回对应模块。所有 `CustomOp` 都同时提供 `forward_native`（可被 `torch.compile`）和 `forward_cuda`（Triton kernel）。

---

## 三、Rotary Position Embedding（RoPE）

[rotary_embedding/](../../vllm/model_executor/layers/rotary_embedding/) 是一个庞大的 RoPE 家族，统一基类是 [base.py](../../vllm/model_executor/layers/rotary_embedding/base.py) 的 `RotaryEmbeddingBase`/`RotaryEmbedding`。

### 3.1 基类机制

- 构造时按 `head_size`、`rotary_dim`（= `head_size * partial_rotary_factor`）、`base`（theta）、`max_position` 预计算 **cos/sin cache**：`inv_freq = 1/(base ** (arange/dim))`，外积位置得到 `cos_sin_cache[max_position, rotary_dim]`（[base.py:83](../../vllm/model_executor/layers/rotary_embedding/base.py#L83)）。
- forward 签名是 `(positions, query, key=None)`：用 `positions.index_select` 从 cache 取对应位置的 cos/sin，对 query/key 的 rotary_dim 部分做旋转（NeoX 风格交错或 GPT-J 风格拆分），**原地**修改 query/key。CUDA 走 `ops.rotary_embedding` 或 FlashInfer 的融合 kernel。
- key 可以为 None（跨层 KV 共享场景，[base.py:167](../../vllm/model_executor/layers/rotary_embedding/base.py#L167)）。

### 3.2 get_rope 工厂与缓存

[__init__.py:36](../../vllm/model_executor/layers/rotary_embedding/__init__.py#L36) 的 `get_rope(...)` 按 `rope_parameters["rope_type"]` 选择子类，并用 `_ROPE_DICT` 按 `(head_size, rotary_dim, max_position, is_neox_style, rope_params, ...)` 做 key **缓存实例**——相同配置的层共享一个 RoPE 模块和 cos/sin cache，省显存。

### 3.3 各变体（按 scaling type）

| 文件 | 类 | 用途 |
|------|-----|------|
| `base.py` | `RotaryEmbedding` | 标准 RoPE（无 extrapolation） |
| `linear_scaling_rope.py` | `LinearScalingRotaryEmbedding` | 线性缩放位置（训练长度内插） |
| `dynamic_ntk_scaling_rope.py` / `dynamic_ntk_alpha_rope.py` | Dynamic NTK | 按上下文长度动态调整 base |
| `ntk_scaling_rope.py` | NTKScaling | 固定 NTK 插值 |
| `yarn_scaling_rope.py` | YaRNScalingRotaryEmbedding | YaRN（Llama 长上下文常用，带温度/幅度修正） |
| `llama3_rope.py` | Llama3RotaryEmbedding | Llama 3 的频率切分 + 缩放 |
| `phi3_long_rope_scaled_rope.py` | Phi3LongRoPEScaled | Phi-3 长上下文 |
| `deepseek_scaling_rope.py` | DeepseekScaling / DeepseekV4Scaling | DeepSeek V2/V4 |
| `mrope.py` / `mrope_interleaved.py` | MRotaryEmbedding | 多维 RoPE（Qwen2-VL：3 个维度段分别计数） |
| `xdrope.py` | XDRotaryEmbedding | 扩展维度 RoPE（HunYuan-VL） |
| `dual_chunk_rope.py` | DualChunkRotaryEmbedding | 双块注意力（DCA）配套 |
| `gemma4_rope.py` | Gemma4RotaryEmbedding | Gemma 4 |
| `fope.py` | FourierRotaryEmbedding | 傅里叶位置编码 |
| `llama4_vision_rope.py`、`ernie45_vl_rope.py`、`telechat3_scaling_rope.py` | 各模型专用 | 视觉/特定模型 |

这些子类主要覆盖 `_compute_inv_freq`/`_compute_cos_sin_cache`（不同的频率缩放策略），forward 多数复用基类。ModelRunner 在 `_calc_mrope_positions`/`_calc_xdrope_positions` 里专门计算 MRoPE/XDRoPE 的位置（见第 04 篇）。

---

## 四、Embedding 与输出头

### 4.1 VocabParallelEmbedding

[vocab_parallel_embedding.py:192](../../vllm/model_executor/layers/vocab_parallel_embedding.py#L192)：词表按 **vocab 维**切到各 TP rank。

- 词表先 padding 到 `pad_vocab_size`（默认 pad 到 64 的倍数，[L81](../../vllm/model_executor/layers/vocab_parallel_embedding.py#L81)），再按 `tp_size` 平分，每 rank 持有 `[vocab/tp, hidden]`。
- forward（[L470](../../vllm/model_executor/layers/vocab_parallel_embedding.py#L470)）：把不在本 rank 范围的 token id 用 `get_masked_input_and_mask` 掩掉，embedding 后 all-reduce 求和——每个 token 只有一个 rank 有非零贡献，reduce 后即得到正确 embedding。
- `VocabParallelEmbeddingShardIndices`（[L104](../../vllm/model_executor/layers/vocab_parallel_embedding.py#L104)）精确处理"原始词表 + padding"两段在不同 tp rank 下的起止索引，供权重加载使用。
- 支持量化 embedding（通过 quant method，如 FP8）和 `org_vocab_size`/`added_vocab_size`（追加的特殊 token）。

### 4.2 ParallelLMHead

[vocab_parallel_embedding.py:503](../../vllm/model_executor/layers/vocab_parallel_embedding.py#L503)：继承 `VocabParallelEmbedding`，作为输出层（logits）。`tie_weights(embed_tokens)`（[L556](../../vllm/model_executor/layers/vocab_parallel_embedding.py#L556)）支持输入 embedding 与输出头**权重共享**。v1 里 logits 计算在 ModelRunner 里通过 `model.compute_logits` 调用，通常只对需要采样的位置算。

---

## 五、Attention 层

### 5.1 Attention：模型看到的注意力层

[attention/attention.py:177](../../vllm/model_executor/layers/attention/attention.py#L177) 是所有 decoder-only 自注意力的统一层。它本身**不直接做注意力计算**，而是：

1. **构造时选定 backend 并创建 impl**：`get_attn_backend(head_size, dtype, kv_cache_dtype, ...)` 选后端（第 03 篇），再 `impl_cls(...)` 创建 `self.impl`（[L344-357](../../vllm/model_executor/layers/attention/attention.py#L344-L357)）；
2. **把自己注册进 `static_forward_context[prefix]`**（[L370](../../vllm/model_executor/layers/attention/attention.py#L370)）：ModelRunner 绑定 KV cache 时就是通过这个字典找到每一层并设置 `.kv_cache`；
3. **持有 `self.kv_cache`**：由 runner 的 `bind_kv_cache` 注入（第 04 篇）；
4. **支持 KV cache 量化**：`_init_kv_cache_quant`、`set_default_quant_scales` 设置 `_k_scale/_v_scale/_q_scale/_prob_scale`（FP8/NVFP4 等），`calculate_kv_scales` 在前向第一次时按 max-abs 动态计算 scale（[L503](../../vllm/model_executor/layers/attention/attention.py#L503)）。

forward（[L409-501](../../vllm/model_executor/layers/attention/attention.py#L409-L501)）数据流：

```python
def forward(query, key, value, output_shape=None):
    if self.calculate_kv_scales:
        torch.ops.vllm.maybe_calc_kv_scales(query, key, value, layer_name)
    output = torch.empty(output_shape or [num_tokens, num_heads*head_size_v])
    query = query.view(-1, num_heads, head_size)
    key   = key.view(-1, num_kv_heads, head_size)
    value = value.view(-1, num_kv_heads, head_size_v)

    # 若 backend 不在 forward 内部写 KV cache（forward_includes_kv_cache_update=False）
    # 则先单独调 unified_kv_cache_update
    if not backend.forward_includes_kv_cache_update:
        unified_kv_cache_update(key, value, layer_name)

    # 调统一注意力算子（custom op，从 forward_context 取 attn_metadata）
    unified_attention_with_output(query, key, value, output, layer_name)
    return output.view(-1, hidden_size)
```

这里的关键解耦：Attention 层从 `get_forward_context().attn_metadata` 取元数据（ModelRunner 在 `set_forward_context` 时塞进去），因此 forward 签名里没有 metadata 参数，对 `torch.compile` 友好。`use_direct_call` 决定是直接调 Python 函数还是走 `torch.ops.vllm.*` 不透明算子（影响编译器处理方式，[L365](../../vllm/model_executor/layers/attention/attention.py#L365)）。

`get_kv_cache_spec`（[L538](../../vllm/model_executor/layers/attention/attention.py#L538)）按 sliding_window/head 配置返回 `FullAttentionSpec`/`SlidingWindowSpec`，这正是 ModelRunner.get_kv_cache_spec 收集的对象。`kv_sharing_target_layer_name` 让该层复用更早层的 KV cache（You Only Cache Once 等）。

### 5.2 MLA：Multi-Head Latent Attention

DeepSeek V2/V3/V4 的 MLA 把 K/V 压缩成低秩潜在向量。两部分：

- **`mla.py` 的 `MultiHeadLatentAttentionWrapper`**（[mla.py:33](../../vllm/model_executor/layers/mla.py#L33)）：外层投影。接收 `MLAModules`（q/kv 的降维投影、RMSNorm、RoPE、o_proj、可选 indexer），forward（[mla.py:131](../../vllm/model_executor/layers/mla.py#L131)）做：
  1. `fused_qkv_a_proj`（或 `kv_a_proj_with_mqa` + `q_proj`）得到低秩 q/kv；
  2. `kv_a_layernorm`/`q_a_layernorm` 归一化；
  3. split 出 `kv_c`（潜在 KV）和 `k_pe`（RoPE 部分），对 q 和 k_pe 应用 RoPE；
  4. 稀疏 MLA 时跑 `indexer` 选 top-k 块；
  5. 交给 `self.mla_attn`（MLAAttention）做注意力；
  6. `o_proj` 输出。
- **`attention/mla_attention.py` 的 `MLAAttention`**（约 91KB）：实际的 MLA 注意力 impl，负责把潜在向量上投影、与 RoPE 部分拼接、调 FlashMLA/FlashAttention-MLA/CUTLASS/Triton MLA 等后端（第 03 篇的 `mla/` 子目录），并管理 absord/absorb 模式（把上投影矩阵吸收进 q 或 kv 以减少计算）。

`MLAModules`（[mla.py:13](../../vllm/model_executor/layers/mla.py#L13)）是个 dataclass，把模型构造的一堆子模块打包传给 wrapper，便于 OOT 后端整体替换。

### 5.3 其他注意力变体层

- **`CrossAttention`**（[cross_attention.py](../../vllm/model_executor/layers/attention/cross_attention.py)）：encoder-decoder 模型的解码器交叉注意力，KV cache 是 encoder 输出（`CrossAttentionSpec`）。
- **`EncoderOnlyAttention`**（[encoder_only_attention.py](../../vllm/model_executor/layers/attention/encoder_only_attention.py)）：双向注意力（ViT 等），不写 KV cache 到主缓存（`EncoderOnlyAttentionSpec`，runner 单独成组）。
- **`MMEncoderAttention`**（[mm_encoder_attention.py](../../vllm/model_executor/layers/attention/mm_encoder_attention.py)）：多模态编码器注意力，与 encoder cache 管理配合。
- **`ChunkedLocalAttention`**（[chunked_local_attention.py](../../vllm/model_executor/layers/attention/chunked_local_attention.py)）：分块局部注意力。
- **`StaticSinkAttention`**（[static_sink_attention.py](../../vllm/model_executor/layers/attention/static_sink_attention.py)）：带 attention sink（固定保留前几个 token）。

### 5.4 线性注意力 / SSM 家族

除了标准 softmax 注意力，layers 下还有几类"注意力状"层（都实现 `AttentionLayerBase`，由各自后端处理）：

- **`lightning_attn.py`**：Lightning Attention，线性注意力的分块实现（带 block-sparse）；
- **`kda.py` 的 `KimiDeltaAttention`**（[kda.py:86](../../vllm/model_executor/layers/kda.py#L86)）：Kimi K2 的 DeltaAttention，线性注意力 + 门控；
- **`mamba/`**：Mamba SSM（详见下节）；
- **`fla/`**：集成第三方 Flash Linear Attention 库的算子（含 gated DeltaNet 等）。

---

## 六、MoE：FusedMoE

[fused_moe/layer.py:219](../../vllm/model_executor/layers/fused_moe/layer.py#L219) 的 `FusedMoE` 是所有 MoE 模型（Mixtral/DeepSeek/Qwen-MoE 等）共用的层。它把 N 个专家的 FFN（每个是 `w1(gate)/w3(up)` + SiLU + `w2(down)`）融合成一个高效算子。

### 6.1 构造与并行策略

构造参数（[L254](../../vllm/model_executor/layers/fused_moe/layer.py#L254)）包括 `num_experts`、`top_k`、`hidden_size`、`intermediate_size`、TP/EP/DP/PCP 并行度、`quant_config`、共享专家等。关键决策：

- **TP vs EP**（[L677-697](../../vllm/model_executor/layers/fused_moe/layer.py#L677-L697)）：默认专家按 TP 切（每 rank 持有所有专家的一部分），开启 EP（expert parallel）时专家分布在不同 rank 上，通过 all-to-all 发送 token。`determine_expert_map` 决定本 rank 持有哪些全局专家，支持 EPLB（专家并行负载均衡，会动态调整专家放置）和 redundant experts。
- **路由器**：内部创建 router（`fused_moe/router/`），支持 softmax/sigmoid 打分、grouped top-k（DeepSeek 风格的"先选组再选专家"）、自定义路由函数、score 修正偏置。
- **quant_method**：和 Linear 一样可插拔，未量化用 `UnquantizedFusedMoEMethod`，各量化格式（FP8/Marlin/GPTQ/AWQ/compressed-tensors 等）提供自己的 fused kernel。
- 把自己注册进 `static_forward_context[prefix]` 和 `static_all_moe_layers`（[L348-349](../../vllm/model_executor/layers/fused_moe/layer.py#L348-L349)），供 ModelRunner 做 EPLB、routed-experts capturer 等。

### 6.2 forward 数据流

层本身的 forward 很薄（[L1554](../../vllm/model_executor/layers/fused_moe/layer.py#L1554)）：`self.runner.forward(hidden_states, router_logits, input_ids)`。真正逻辑在 [runner/moe_runner.py](../../vllm/model_executor/layers/fused_moe/runner/moe_runner.py) 的 `MoERunner`，经典步骤：

```
hidden_states [num_tokens, hidden]
   │ 1. router_logits = gate(hidden_states)        # [num_tokens, num_experts]
   │ 2. top-k 路由 → topk_weights [num_tokens, k], topk_ids [num_tokens, k]
   │ 3. （EP 时）按目标专家做 all-to-all，把 token 发到对应 rank
   │ 4. moe_align_block_size：按专家分组 token，生成 permutation 索引
   │ 5. 对每个专家：
   │      w13 = concat(w1, w3)                    # 融合 gate/up
   │      x_permuted @ w13 → silu_and_mul → @ w2  # 专家 FFN
   │    （由 experts/ 下的 cutlass/deepgemm/triton/flashinfer/marlin 等 fused kernel 完成）
   │ 6. 按 topk_weights 加权求和各专家输出
   │ 7. （EP 时）all-to-all 把结果发回源 rank，反 permute
   ▼
output [num_tokens, hidden]
```

`fused_moe/experts/` 是各硬件/量化的专家 GEMM 实现：`cutlass_moe.py`、`deep_gemm_moe.py`、`triton_moe.py`、`flashinfer_cutlass_moe.py`、`marlin_moe.py`、`trtllm_*_moe.py`、`rocm_aiter_moe.py`、`cpu_moe.py`、`xpu_moe.py` 等。`fused_moe.py` 里的 Triton kernel（`fused_moe_kernel`/`invoke_fused_moe_triton_kernel`，[L295](../../vllm/model_executor/layers/fused_moe/fused_moe.py#L295)）是参考实现和兜底。

### 6.3 共享专家与辅助组件

- **共享专家**：`shared_experts`（`runner/shared_experts.py` 的 `SharedExperts`）是一个始终作用于所有 token 的 FFN，输出加到路由专家结果上（DeepSeek V3 等）。ROCm AITER 支持把共享专家与路由专家融合。
- **`prepare_finalize/`**：token permutation/unpermutation、all-to-all 前后的布局变换。
- **`all2all_utils.py`**：EP 的 all-to-all 通信封装。
- **`routed_experts_capturer.py`**：调试/可观测用，捕获每步每个 MoE 层路由到了哪些专家（对应 Worker 的 `enable_return_routed_experts`）。
- **`lora_experts_mixin.py`/`lora_context.py`**：MoE + LoRA（专家级 LoRA 适配）。
- **EPLB**：`set_eplb_state`/`update_expert_map`/`ensure_round_robin_expert_routing_tables` 在运行时重排专家到 rank 的映射以均衡负载（第 04 篇 `eplb_step` 调用）。

权重加载非常复杂（[L1088-1446](../../vllm/model_executor/layers/fused_moe/layer.py#L1088-L1446)）：checkpoint 里专家权重是 `experts.N.w1/w2/w3`，`weight_loader` 要处理 TP 切分、w13 融合、grouped/per-channel scale、g_idx（GPTQ）、zero experts、冗余专家逻辑到物理映射（EPLB）等。`make_expert_params_mapping`（[L1572](../../vllm/model_executor/layers/fused_moe/layer.py#L1572)）生成统一的 `(param_name, weight_name, expert_id, shard_id)` 映射表。

---

## 七、Mamba 与状态空间模型

[mamba/](../../vllm/model_executor/layers/mamba/) 实现 SSM 类混合模型，它们和 Attention 一样是"有状态"层（需要 recurrent state 作为 KV cache），所以基类 `MambaBase` 也继承 `AttentionLayerBase`（[abstract.py:16](../../vllm/model_executor/layers/mamba/abstract.py#L16)），提供 `get_kv_cache_spec`/`get_attn_backend`/`get_state_shape`。

### 7.1 MambaMixer（Mamba-1）

[mamba_mixer.py:51](../../vllm/model_executor/layers/mamba/mamba_mixer.py#L51)：选择性状态空间模型。`forward_impl`（[L239](../../vllm/model_executor/layers/mamba/mamba_mixer.py#L239)）步骤：

1. `in_proj` 把输入投影成 `(BC, gate)` 两部分；
2. 用 `split_batch_to_prefill_and_decode` 把 prefill token 和 decode token 分开（两者走不同 kernel）；
3. **卷积阶段**：`causal_conv1d` 对 BC 做短时卷积，状态存在 conv_state KV cache 里；
4. **SSM 阶段**：`ssd_combined`/`mamba_ssm` 做选择性扫描（selective scan），产生上下文表示，状态存在 ssm_state KV cache；
5. 乘以 `silu(gate)`，再 `out_proj` 回 hidden 维。

KV cache 有两个张量（`self.kv_cache[0]` conv_state、`[1]` ssm_state），`get_state_shape` 声明它们的形状。mamba cache mode（`all`/`align`）决定状态是每请求独立分配还是按块共享/拷贝（与第 04 篇 `preprocess_mamba` 对应）。

### 7.2 MambaMixer2（Mamba-2 / SSD）

[mamba_mixer2.py:231](../../vllm/model_executor/layers/mamba/mixer_mixer2.py)：Mamba-2 把 SSM 写成**状态空间对偶（SSD）**形式——可化为半可分矩阵的注意力状计算，prefill 用 chunk-wise 的 SSD 并行算法、decode 用递推。文件内有 `Mixer2RMSNormGated`（融合门控 RMSNorm），forward（[L519](../../vllm/model_executor/layers/mamba/mamba_mixer2.py#L519)）和 `conv_ssm_forward`（[L560](../../vllm/model_executor/layers/mamba/mamba_mixer2.py#L560)）协调 conv1d、SSD、x/y/z 投影、DT 投影、A/B/C 参数。

### 7.3 算子与其他 SSM

- **`ops/`**：`causal_conv1d`（因果 1D 卷积，Triton/C++）、`ssd_bmm/ssd_chunk_scan/ssd_chunk_state/ssd_combined/ssd_state_passing`（SSD 的各阶段 Triton kernel）、`ssu_dispatch.py`（在不同 SSM 实现间分发）、`layernorm_gated`（融合门控 LayerNorm）、`cpu/`（CPU 后端的 conv1d、gdn、recurrent gated delta rule）。
- **`gdn_linear_attn.py`**：Gated DeltaNet 线性注意力。
- **`linear_attn.py`/`short_conv.py`**：通用线性注意力、短卷积混合块。
- **`lamport_workspace.py`**：混合模型里 attention/SSM 之间共享的临时 workspace。

---

## 八、池化头（Pooler）

[pooler/](../../vllm/model_executor/layers/pooler/) 服务于 embedding/reward/classification 等非生成模型。抽象基类 `Pooler`（[abstract.py:16](../../vllm/model_executor/layers/pooler/abstract.py#L16)）声明 `get_supported_tasks()`、`get_pooling_updates(task)`（把 `PoolingParams` 的设置应用进来，如 prompt 长度、附加 token）、`forward(...)`。

两类：

- **`seqwise/`（序列级）**：`SequencePooler`（[seqwise/poolers.py:44](../../vllm/model_executor/layers/pooler/seqwise/poolers.py#L44)）对整条序列生成一个向量，配 `EmbeddingPoolerHead`（归一化/投影成 embedding）或 `ClassifierPoolerHead`（分类/打分）。包含 LAST/AVG/CLAP 等聚合方式，以及 `PoolerNormalize`/`PoolerClassify`/`PoolerMultiLabelClassify` 激活。
- **`tokwise/`（token 级）**：`TokenPooler` 对每个 token 输出（如 token 分类、late-interaction ColBERT），配 `TokenEmbeddingPoolerHead`/`TokenClassifierPoolerHead`。
- **`special.py`**：模型专用池化——`BgeM3Pooler`（BGE-M3 的稠密+稀疏+多向量）、`BOSEOSFilter`、`DispatchPooler`（按任务分派）、`IdentityPooler`。

ModelRunner 的 `_pool`/`late_interaction_runner` 在请求结束时调用这些 pooler，结果通过 `pooler_output` 返回（第 04 篇）。

---

## 九、多模态与融合专用层

- **`resampler.py`**：共享的 perceiver resampler（Flamingo 风格），把变长视觉特征压缩成固定数量的 KV token，供 Qwen-VL/IDEFICS 等使用；包含 `Qformer`/线性投影/注意力等。
- **`conv.py`**：`Conv2dLayer`/`Conv3d` 等视觉模型卷积，继承 `CustomOp` 做设备分派，带权重的 padding/TP。
- **`mhc.py`**："mHC"（MoE-Hybrid Compute）深度融合块，把多个 GEMM、norm、激活融合成少数 kernel（NVIDIA 优化路径），有 pre/post block 与 GEMM FMA。
- **`deepseek_v4_attention.py` + `deepseek_compressor.py` + `sparse_attn_indexer.py`**：DeepSeek V4 的稀疏 MLA 三件套——indexer 选块、compressor 压缩/量化 KV cache 并插入稀疏注意力、主注意力层调稀疏算子。
- **`lightning_attn.py`/`kda.py`**：前述线性注意力。
- **`fla/`**：Flash Linear Attention 集成。

---

## 十、两条横切机制

### 10.1 CustomOp 与 PluggableLayer

- **`CustomOp`**（[custom_op.py:103](../../vllm/model_executor/custom_op.py#L103)）：按当前设备分派到 `forward_native/cuda/hip/xpu/cpu/tpu/oot`。`forward_native` 用 `ir.ops.*`（中间表示）写，既可被 `torch.compile` 编译，也可降级到各后端 kernel；`forward_cuda` 通常是手写 Triton。`enabled()` 类方法允许通过 env 全局开关某个 op。
- **`PluggableLayer`**（[custom_op.py:32](../../vllm/model_executor/custom_op.py#L32)）：`__new__` 里检查是否有 OOT（out-of-tree）后端用 `@PluggableLayer.register(name)` 或 `register_oot` 替换了该层，若有则返回替换类的实例。这让 vLLM 能在不改核心代码的情况下整体替换 Linear/Attention/MoE 等层（如华为/昇腾、第三方加速器）。`@PluggableLayer.register("replicated_linear")` 这样的装饰器给层起规范名，OOT 后端按名替换。

### 10.2 Batch Invariant

[batch_invariant.py](../../vllm/model_executor/layers/batch_invariant.py) 提供一批"与 batch 形状无关"的持久化 Triton kernel（`matmul_persistent`、`bmm`、`rms_norm`、`softmax`、`mean`、`linear_batch_invariant` 等）。当 `VLLM_BATCH_INVARIANT` 开启时，Linear/RMSNorm 等会走这些 kernel。它们的目标是：**同样的编译产物在不同 batch/seq 长度下都能高效运行**，避免为每种形状重新编译或出现性能悬崖，对 CUDA graph + 可变 batch 场景友好。`init_batch_invariance()` 在 worker 分布式初始化时调用（第 04 篇）。

---

## 小结

`model_executor/layers/` 可以看作"模型无关的神经网络积木箱"：

1. **Linear/Embedding 把张量并行和量化做成透明组合**——模型代码只写 `ColumnParallelLinear`/`RowParallelLinear`，TP 切分、all-gather/all-reduce、权重量化/反量化全部由层和 quant_method 处理；
2. **Attention 层是薄壳**：构造时选 backend 并注册到 `static_forward_context`，forward 从 `forward_context` 取 metadata，调用统一的 `unified_attention` custom op，KV cache 由 runner 注入——这正是第 03、04 篇能把"注意力后端"和"worker"独立出去的原因；
3. **RoPE/Norm/Activation 是 `CustomOp` 设备分派 + native 可编译实现**的典范，同一份逻辑在 CUDA/CPU/ROCm 上各走最优 kernel；
4. **FusedMoE 是最复杂的层**：内置 TP/EP 并行、top-k/grouped 路由、token permutation、各硬件 fused GEMM、共享专家、EPLB 负载均衡、量化和 LoRA 扩展，但对模型只暴露 `forward(hidden, router_logits)`；
5. **Mamba/线性注意力与 Attention 实现同一接口**（`AttentionLayerBase`），因此能在混合模型里与标准注意力共用 KV cache 管理、调度和注意力后端体系；
6. **PluggableLayer/CustomOp + batch_invariant** 保证整套层可被 OOT 后端替换、可跨平台、可在动态 shape 下稳定高性能运行。

下一篇 [06 模型实现](./06_model_implementations.md) 将进入 `vllm/model_executor/models/`，看具体模型（Llama/Qwen/DeepSeek 等）如何用本篇这些积木搭出完整网络，以及 `ModelRegistry`/`Supports*` 接口/权重加载的约定。
