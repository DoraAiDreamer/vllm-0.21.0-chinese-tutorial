# 模型实现

> 源码路径: `vllm/model_executor/models/`

这是 vLLM 最大的一个目录（290+ 文件），每个文件对应一个或多个模型架构（Llama、Qwen、DeepSeek、Gemma、Mixtral、Mamba、各种 ViT…）。它们用第 05 篇的"积木"（Linear、Attention、FusedMoE、RoPE、RMSNorm、Embedding…）搭出具体网络。本篇**不逐个列模型**，而是讲清楚所有模型共同遵守的契约：

1. 模型怎么被**注册与解析**（`ModelRegistry`、架构名字符串→类）；
2. 一个模型类必须实现什么（`VllmModel` 接口、`Supports*` 能力协议）；
3. 模型的标准结构与 forward 数据流（以 Llama/Qwen 为参照）；
4. 权重加载约定（`load_weights`、`weight_loader`、packed weights、PP missing）；
5. 多模态、MoE、混合模型（hybrid）、池化模型的扩展点；
6. 辅助工具（`make_layers`、`AutoWeightsLoader`、adapter 装饰器）。

理解了这些约定，再读任何一个具体模型文件都是"套模板 + 处理该模型的特殊结构"。

---

## 一、模型注册表

### 1.1 从 architectures 字符串到模型类

Hugging Face 的 `config.json` 里有 `"architectures": ["LlamaForCausalLM"]`。vLLM 用这个字符串找到模型类，核心是 [registry.py](../../vllm/model_executor/models/registry.py) 的 `ModelRegistry` 单例（[registry.py:1319](../../vllm/model_executor/models/registry.py#L1319)）。

注册表是一个 dict：`model_arch (str) -> _BaseRegisteredModel`。条目分两类信息源：

- **内置模型表**（[registry.py:70-671](../../vllm/model_executor/models/registry.py#L70-L671)）：按任务类型分成若干 dict：
  - `_TEXT_GENERATION_MODELS`（decoder-only 生成模型，最大的一块）；
  - `_EMBEDDING_MODELS` / `_LATE_INTERACTION_MODELS` / `_REWARD_MODELS` / `_TOKEN_CLASSIFICATION_MODELS` / `_SEQUENCE_CLASSIFICATION_MODELS`（池化/评分模型）；
  - `_MULTIMODAL_MODELS`（视觉/音频等多模态）；
  - `_SPECULATIVE_DECODING_MODELS`（draft 模型）；
  - `_TRANSFORMERS_SUPPORTED_MODELS` / `_TRANSFORMERS_BACKEND_MODELS`（直接走 transformers 后端的模型）。
  
  每个条目形如 `"LlamaForCausalLM": ("llama", "LlamaForCausalLM")`——**模块名 + 类名**，而不是直接的类对象。
- **`_VLLM_MODELS`**（[registry.py:671](../../vllm/model_executor/models/registry.py#L671)）合并上面所有表；
- **`_PREVIOUSLY_SUPPORTED_MODELS`**：曾支持但已移除的架构，给出最后支持的 vLLM 版本（给用户友好报错）；
- **`_OOT_SUPPORTED_MODELS`**：移到外置插件的架构，给出插件 URL。

外部模型可用 `ModelRegistry.register_model(arch, cls_or_str)`（[registry.py:939](../../vllm/model_executor/models/registry.py#L939)）注册，cls 可以是类或 `"<module>:<class>"` 形式的字符串（**延迟导入**，避免在 fork 子进程前初始化 CUDA）。

### 1.2 延迟加载与子进程探测

- **`_LazyRegisteredModel`**（[registry.py:796](../../vllm/model_executor/models/registry.py#L796)）：只存模块名/类名，真正 `load_model_cls()` 时才 import。
- **`_ModelInfo`**（[registry.py:710](../../vllm/model_executor/models/registry.py#L710)）：从模型类上抽取元信息（是否支持 PP、是否多模态、是否 hybrid、runner_type 等）。为了避免 import 重型模型类带来的副作用，`_try_inspect_model_cls` 会在**子进程**里导入并 inspect（`_run_in_subprocess`，[registry.py:1332](../../vllm/model_executor/models/registry.py#L1332)），结果用文件缓存（`_get_cache_dir`）。
- 解析入口 `resolve_model_cls(architecture)`（[registry.py:1176](../../vllm/model_executor/models/registry.py#L1176)）：先在注册表里找，找不到再尝试 `_try_resolve_transformers`（transformers 后端兜底），都失败则调 `_raise_for_unsupported` 给出"不支持/已移除/需装插件"的明确错误。
- 一系列 `is_*_model` 查询（[registry.py:1230-1310](../../vllm/model_executor/models/registry.py#L1230-L1310)）：`is_text_generation_model`/`is_pooling_model`/`is_multimodal_model`/`is_pp_supported_model`/`is_hybrid_model`/`is_attention_free_model`/`has_inner_state` 等，供引擎与 runner 决定走哪条执行路径。

模型加载器（`vllm/model_executor/model_loader/`，下一篇详述）拿到解析出的类后，用 `model_cls(vllm_config=vllm_config, prefix=...)` 实例化。

---

## 二、模型接口契约

### 2.1 VllmModel：所有模型的最小接口

[interfaces_base.py:47](../../vllm/model_executor/models/interfaces_base.py#L47) 的 `VllmModel(Protocol)` 只要求三件事：

```python
class VllmModel(Protocol[T_co]):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(self, input_ids, positions) -> T_co: ...
```

- **构造签名必须含 `vllm_config`**（`is_vllm_model` 用 `supports_kw` 反射检查，[interfaces_base.py:59](../../vllm/model_executor/models/interfaces_base.py#L59)）。模型从 `vllm_config.model_config.hf_config` 读 HuggingFace 配置，从 `cache_config/quant_config/parallel_config/lora_config` 读其余配置。
- **`embed_input_ids`**：把 token id 变成 embedding。多模态模型会重写它以融合视觉嵌入（见第 04 篇 `_preprocess` 调用 `model.embed_input_ids`）。
- **`forward(input_ids, positions, ...)`**：返回隐状态，或 PP 中间的 `IntermediateTensors`，或（EAGLE-3）`(hidden, aux_hidden)`。

两个特化协议：

- **`VllmModelForTextGeneration`**（[L114](../../vllm/model_executor/models/interfaces_base.py#L114)）：增加 `compute_logits(hidden_states) -> logits | None`（TP rank>0 返回 None）。所有生成模型实现它。
- **`VllmModelForPooling`**（[L148](../../vllm/model_executor/models/interfaces_base.py#L148)）：带 `is_pooling_model=True` 类变量、`pooler: Pooler`、默认池化类型/attn_type/score_type。embedding/reward/classification 模型实现它。

`is_vllm_model`/`is_text_generation_model`/`is_pooling_model` 是运行时类型检查（`@runtime_checkable` Protocol），用于引擎决定 runner 类型与任务支持。

### 2.2 Supports* 能力协议

[interfaces.py](../../vllm/model_executor/models/interfaces.py) 用一组 `Protocol` 声明可选能力，runner/引擎通过 `supports_xxx(model)` 检测并启用对应路径。重要的有：

| 协议 | 要求的关键方法/属性 | 启用的能力 |
|------|---------------------|-----------|
| `SupportsMultiModal`（[L94](../../vllm/model_executor/models/interfaces.py#L94)） | `supports_multimodal=True`、`get_placeholder_str`、`embed_multimodal(...)`、`get_language_model()`、`configure_mm_token_handling` | 多模态输入处理、encoder cache |
| `SupportsMultiModalPruning`（[L411](../../vllm/model_executor/models/interfaces.py#L411)） | `recompute_mrope_positions(...)` | 多模态 token 剪枝 |
| `SupportsLoRA`（[L536](../../vllm/model_executor/models/interfaces.py#L536)） | `packed_modules_mapping`、`embedding_modules`、`embedding_padding_modules`、`targeted_*` 等类变量 | LoRA 适配（第 09 篇） |
| `SupportsPP`（[L614](../../vllm/model_executor/models/interfaces.py#L614)） | `make_empty_intermediate_tensors(...)`、`forward(... intermediate_tensors ...)` | 流水并行 |
| `SupportsMRoPE` / `SupportsXDRoPE` | `mrope_position_delta` 等 | 多维旋转位置编码 |
| `HasInnerState`（[L734](../../vllm/model_executor/models/interfaces.py#L734)） | `get_empty_preload_context`/`get_non_preload_context`/... | 有内部状态的模型（如 Medusa 的头状态） |
| `IsAttentionFree`（[L760](../../vllm/model_executor/models/interfaces.py#L760)） | — | 无注意力模型（Mamba 等），跳过 KV cache |
| `IsHybrid`（[L787](../../vllm/model_executor/models/interfaces.py#L787)） | `get_mamba_state_shape_from_config`、`get_mamba_state_copy_func` | 注意力+SSM 混合（Jamba、Qwen3.5-MoE 等） |
| `MixtureOfExperts`（[L844](../../vllm/model_executor/models/interfaces.py#L844))) | `set_eplb_state`、`update_physical_experts_metadata`、`set_moe_parameters` | MoE + EPLB 专家负载均衡 |
| `SupportsTranscription` | ASR 相关 | 语音转录模型 |
| `SupportsEagle`/`SupportsEagle3`（在 eagle mixin） | `get_input_embeddings`/`get_output_embeddings`/... | EAGLE 投机解码所需的隐藏状态访问 |

多数这些协议是**结构性类型**（duck typing + `@runtime_checkable`），模型只要有对应方法/属性即被认定支持，不必显式继承；也常用 mixin（如 `EagleModelMixin`、`QwenNextMixtureOfExperts`）提供默认实现。

---

## 三、标准模型结构（以 Llama 为例）

[llama.py](../../vllm/model_executor/models/llama.py) 是最干净的 decoder-only 模板，绝大多数自回归模型结构与之同构。

### 3.1 三个层次

```
LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3)
   └── self.model = LlamaModel(nn.Module, EagleModelMixin)
          ├── embed_tokens : VocabParallelEmbedding      # 首 PP rank 才有
          ├── layers = ModuleList([LlamaDecoderLayer])  # 按 PP 区间只建本 rank 的层
          │     ├── input_layernorm : RMSNorm
          │     ├── self_attn : LlamaAttention          # 内部包 Attention 层 + qkv/o/o_proj
          │     ├── post_attention_layernorm : RMSNorm
          │     └── mlp : LlamaMLP                      # gate/up(gate_up_proj) + act + down
          └── norm : RMSNorm                            # 末 PP rank 才有
```

- **`LlamaMLP`**（[llama.py:81](../../vllm/model_executor/models/llama.py#L81)）：`gate_up_proj = MergedColumnParallelLinear`（把 gate 和 up 打包成一次 GEMM），`act_fn = SiluAndMul()`，`down_proj = RowParallelLinear`。forward `down_proj(act_fn(gate_up_proj(x)))`。
- **`LlamaAttention`**（[L124](../../vllm/model_executor/models/llama.py#L124)）：`qkv_proj = QKVParallelLinear`（打包 Q/K/V，按 GQA 头数切），`o_proj = RowParallelLinear`，持有一个 `Attention(num_heads, num_kv_heads, ...)` 层。forward 投影出 q/k/v、reshape 成 `[num_tokens, heads, head_dim]`，调用 `self.attn(q, k, v)`——Attention 层再从 forward context 取 metadata 完成计算与 KV 写入。RoPE 由 `_init_rotary_emb` 创建并在 q/k 进 attention 前应用。
- **`LlamaDecoderLayer.forward`**（[L316](../../vllm/model_executor/models/llama.py#L316)）采用**融合 add-norm** 风格：
  ```python
  if residual is None:
      residual = hidden_states
      hidden_states = self.input_layernorm(hidden_states)
  else:
      hidden_states, residual = self.input_layernorm(hidden_states, residual)  # 融合 add+norm
  hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
  hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
  hidden_states = self.mlp(hidden_states)
  return hidden_states, residual
  ```
  RMSNorm 支持接收 residual 返回 `(normed, residual)`，把残差加与归一化融合，减少 kernel 启动和显存读写。

### 3.2 LlamaModel.forward 的数据流

[llama.py:395-434](../../vllm/model_executor/models/llama.py#L395-L434)：

```python
def forward(self, input_ids, positions, intermediate_tensors=None,
            inputs_embeds=None, **extra_layer_kwargs):
    if get_pp_group().is_first_rank:
        hidden_states = inputs_embeds if (inputs_embeds is not None) \
                                      else self.embed_input_ids(input_ids)
        residual = None
    else:
        hidden_states = intermediate_tensors["hidden_states"]
        residual    = intermediate_tensors["residual"]

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer in islice(self.layers, self.start_layer, self.end_layer):
        hidden_states, residual = layer(positions, hidden_states, residual, **extra_layer_kwargs)
        self._maybe_add_hidden_state(aux_hidden_states, idx+1, ...)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": ..., "residual": ...})

    hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states: return hidden_states, aux_hidden_states   # EAGLE-3
    return hidden_states
```

要点：
- **PP 边界**：首 rank 做 embedding，非首 rank 从 `intermediate_tensors` 取；非末 rank 返回 `IntermediateTensors` 给下一 stage；末 rank 做最终 norm 后返回隐状态。这正好对应第 04 篇 Worker 的 PP 收发逻辑。
- **只遍历本 rank 的层区间** `[start_layer, end_layer)`，区间外是 `PPMissingLayer`（占位 Identity，[utils.py:607](../../vllm/model_executor/models/utils.py#L607)），保证 `state_dict` 名字与全模型一致但不占显存。
- **EAGLE-3 辅助隐状态**：`_maybe_add_hidden_state` 按配置收集若干中间层输出，供 draft model 使用。

### 3.3 LlamaForCausalLM

[llama.py:501](../../vllm/model_executor/models/llama.py#L501) 是最外层：

- 类属性 `packed_modules_mapping`（告诉 LoRA/权重加载 qkv_proj 由 q/k/v 打包、gate_up_proj 由 gate/up 打包）、`embedding_modules`（LoRA 可适配 embedding）。
- `__init__` 里 `self.model = LlamaModel(...)`，若是末 PP rank 才建 `self.lm_head = VocabParallelEmbedding`（或 `ParallelLMHead`），支持 `tie_weights`。
- `forward(input_ids, positions, intermediate_tensors, ...)` 直接 `return self.model(...)`。
- `compute_logits(hidden_states)` 仅在末 rank：`logits = self.lm_head.logits_processor(...)`，用 `LogitsProcessor` 做张量并行的 logits 计算（嵌入权重当输出矩阵乘）。
- `load_weights(weights)` 见下文。
- `embed_input_ids` 转发给 `self.model.embed_input_ids`。

> 注意：`LlamaBidirectionalForSequenceClassification`/`LlamaBidirectionalModel` 用 `as_embedding_model(LlamaForCausalLM)` / `as_seq_cls_model(...)` 适配器（[adapters.py:230](../../vllm/model_executor/models/adapters.py#L230)）从同一个生成类派生池化/分类模型，避免重复代码。

---

## 四、权重加载约定

这是模型实现里最"体力活"也最关键的部分。机制是分层的：

### 4.1 三层加载钩子

1. **每个参数的 `weight_loader`**：`set_weight_attrs` 在参数上挂一个可调用对象（见第 05 篇 vLLMParameter）。Linear/Embedding/MoE 各自定义了按 TP 切分、打包融合矩阵、处理量化 scale 的加载逻辑。
2. **每个子模块的 `load_weights(weights)`**：模型/层可以重写它来自定义遍历逻辑。
3. **`AutoWeightsLoader`**（[utils.py:117](../../vllm/model_executor/models/utils.py#L117)）：通用加载器，自动遍历模块树、对每个权重找对应参数并调其 `weight_loader`；如果子模块自己有 `load_weights` 就交给它。模型一般在自己的 `load_weights` 里做 checkpoint 名→vLLM 名的 remap，再复用 `AutoWeightsLoader` 或显式循环。

### 4.2 典型 load_weights 做的事

看 Llama（[llama.py:436-498](../../vllm/model_executor/models/llama.py#L436-L498)）和 Qwen3.5（[qwen3_5.py:274](../../vllm/model_executor/models/qwen3_5.py#L274)、[L518](../../vllm/model_executor/models/qwen3_5.py#L518)），通用步骤：

1. **跳过无关权重**：如 `rotary_emb.inv_freq`、`cos_cached/sin_cached`（运行时重算）。
2. **融合矩阵重映射**（`stacked_params_mapping`）：checkpoint 里是分开的 `q_proj/k_proj/v_proj`，要写进融合的 `qkv_proj`；`gate_proj/up_proj` 写进 `gate_up_proj`。对每个权重，把名字替换成融合名，带上 `shard_id`（"q"/"k"/"v" 或索引），调 `param.weight_loader(param, loaded_weight, shard_id)`。
3. **量化 scale/zero-point 重映射**：调 `quant_config.get_cache_scale(name)` 或 `maybe_remap_kv_scale_name`（FP8 KV cache）。
4. **PP missing 跳过**：`is_pp_missing_parameter(name, self)` 判断该参数在当前 PP rank 是否不存在（是别的 stage 的层），跳过。
5. **MoE 专家权重重排**：如 Qwen3.5 MoE 的 `load_fused_expert_weights`（[qwen3_5.py:248](../../vllm/model_executor/models/qwen3_5.py#L248)）把 checkpoint 的专家权重融合成 w13/w2 排布；`get_expert_mapping`（[L537](../../vllm/model_executor/models/qwen3_5.py#L537)）提供逻辑专家→物理专家的映射（EPLB/冗余专家）。
6. **权重名前缀处理**：多模态模型语言模型可能带 `model.language_model.` 前缀，用 `maybe_prefix(prefix, name)` 处理。
7. 返回已加载参数名集合，供加载器核对是否有遗漏/多余。

### 4.3 辅助工具

- **`WeightsMapper`**（[utils.py:44](../../vllm/model_executor/models/utils.py#L44)）：声明式地做权重名映射/丢弃/子串替换，模型用它处理 HF 命名差异（如某些模型把 `c_attn` 拆成 qkv）。
- **`default_weight_loader`**：最朴素的拷贝（处理 shard/量化）。
- **`make_layers`**（[utils.py:620](../../vllm/model_executor/models/utils.py#L620)）：构造层列表时按 PP rank 用 `get_pp_indices` 算 `[start, end)`，区间内建真实层、区间外填 `PPMissingLayer`，并经过 offloader 包装（CPU offload）。
- **`make_empty_intermediate_tensors_factory(keys, hidden_size)`**（[utils.py:688](../../vllm/model_executor/models/utils.py#L688)）：生成 PP 用的空中间张量（零张量占位）。
- **`collect_children`/`flatten_bn`**：处理嵌套/列表型参数（专家权重）。
- **`no_init_weights`**：构造模型时避免真实初始化权重（反正会从 checkpoint 覆盖），加速且省显存。

---

## 五、多模态模型

多模态模型实现 `SupportsMultiModal`，并通过 `MULTIMODAL_REGISTRY` 注册输入处理器（第 10 篇详述）。模型侧的关键点：

- **占位符与嵌入合并**：prompt 里有图片占位 token（如 `<|image_pad|>`），模型在 `embed_input_ids`/`embed_multimodal` 里把视觉编码器产出的 embedding 按 `PlaceholderRange` 散布到对应位置，与文本 embedding 合并。第 04 篇 `_gather_mm_embeddings` + `_merge_multimodal_embeddings`（[utils.py:458](../../vllm/model_executor/models/utils.py#L458)）完成散布。
- **`get_language_model()`**：返回内部纯语言模型，让 runner 能绕过视觉塔直接拿到 transformer、挂 KV cache、做 EAGLE 等。interfaces 提供 `_mark_language_model`/`_mark_tower_model`/`_mark_composite_model` 上下文（[interfaces.py:213-322](../../vllm/model_executor/models/interfaces.py#L213-L322)）标记哪些子模块是语言模型/视觉塔/复合结构。
- **`embed_input_ids` 被重写**：例如 Qwen3.5-VL 的条件生成类 `Qwen3_5ForConditionalGeneration.embed_input_ids`（[qwen3_5.py:590](../../vllm/model_executor/models/qwen3_5.py#L590)）区分纯文本与带视觉输入。
- **MRoPE 重算**：`recompute_mrope_positions`（[qwen3_5.py:616](../../vllm/model_executor/models/qwen3_5.py#L616)）在多模态 token 被剪枝时重新计算位置。
- **视觉编码器本身**通常是 encoder-only attention（`AttentionType.ENCODER_ONLY`/`ENCODER`），单独有 encoder cache，由 runner 的 `EncoderRunner` 执行（第 04 篇）。
- **`ProcessingInfo` 子类**（如 [qwen3_5.py:108](../../vllm/model_executor/models/qwen3_5.py#L108) 的 `Qwen3_5ProcessingInfo`）提供每个模型的 H×W→token 数、占位符数量等给多模态 registry。

---

## 六、MoE 与混合模型

### 6.1 MoE 模型

实现 `MixtureOfExperts` 协议的模型（如 Qwen3-MoE、DeepSeek）在层里用第 05 篇的 `FusedMoE`：

- 构造 `gate = ReplicatedLinear`（router，所有 TP rank 都有）+ `experts = FusedMoE(...)`；
- forward `router_logits = gate(hidden_states); hidden_states = self.experts(hidden_states, router_logits)`；
- 提供 `packed_modules_mapping` 给 LoRA，提供 expert 映射给权重加载与 EPLB；
- `set_moe_parameters`/`update_physical_experts_metadata`（如 [qwen3_5.py:727](../../vllm/model_executor/models/qwen3_5.py#L727)）在 EPLB 重平衡时更新路由表与物理专家元数据；
- 模型类常带 `QwenNextMixtureOfExperts` 这类 mixin 提供通用 MoE 逻辑。

### 6.2 混合模型（Hybrid: Attention + SSM）

Jamba、Qwen3.5-MoE-Conditional 等在同一网络里交替 Attention 层和 Mamba 层。它们：

- 继承 `IsHybrid` 协议，提供 `get_mamba_state_shape_from_config`（[qwen3_5.py:684](../../vllm/model_executor/models/qwen3_5.py#L684)）和 `get_mamba_state_copy_func`（[L717](../../vllm/model_executor/models/qwen3_5.py#L717)），告诉 worker SSM 状态的形状和在块间拷贝的方法（配合第 04 篇 `preprocess_mamba` 与 mamba cache mode）；
- decoder layer 里根据层类型选择 `self_attn`（Attention）或 `mamba`（MambaMixer/MambaMixer2）；
- runner 把 Mamba 层也当作 `AttentionLayerBase`，用各自的 attention backend（mamba/mamba2/gdn 后端，见第 03 篇）产出 KV cache spec 与 metadata。

### 6.3 线性注意力 / 无注意力模型

`IsAttentionFree` 标记无注意力模型（纯 Mamba/SSM），引擎据此跳过标准 KV cache 路径（但仍有 SSM state）。KDA、Lightning Attention、GDN 等模型用第 05 篇的对应层，同样实现 `AttentionLayerBase`。

---

## 七、其他公共模式

- **`logits_processor`**：[logits_processor.py](../../vllm/model_executor/models/logits_processor.py) 定义模型内置 logits 处理基类（如某些模型的 logits softcap、尺度变换、step-wise bias）。模型可返回一个 `LogitsProcessor`，在 sampler 之前对 logits 做变换（`scale`/`shaped_softcap`/`apply_bias`/`fetch`）。它与第 04 篇的采样 logits processors 不同：这是模型自带的、知道模型结构的处理。
- **`config.py` 的 `VerifyAndUpdateConfig`**：各模型（或适配器，如 `SequenceClassificationConfig`）实现一个配置校验/补全钩子，在 engine 初始化时根据任务类型修改 `vllm_config`。
- **`resampler.py`/`conv.py`**：多模态/视觉模型用的 perceiver resampler 与卷积层（第 05 篇）。
- **Eagle/Draft 模型**：`EagleModelMixin`/`Eagle3ModelMixin`、`spec_decode/` draft 模型实现，提供第 04 篇 drafter 所需的隐藏状态抽取接口（`get_eagle3_default_aux_hidden_state_layers`、`set_aux_hidden_state_layers`）。
- **任务适配器**：`adapters.py` 的 `as_embedding_model`/`as_seq_cls_model`/`as_reward_model` 把一个生成模型类包装成池化/分类模型，自动加 pooler、改 `runner_type`、处理配置校验。
- **`score_type` 装饰器/属性**：声明模型是 bi-encoder（embed）、cross-encoder（score/classify）还是 late-interaction（token embed），供 Score API 分派。

---

## 八、一个新模型要写什么（心智模型）

综合来看，在 vLLM 里新增一个 decoder-only 模型大致需要：

1. **一个模型文件**，定义 `XXXModel`（层堆叠 + embedding/norm/PP 处理）和 `XXXForCausalLM`（组装 + `compute_logits` + `load_weights`），用第 05 篇的层搭网络；
2. 在 `registry.py` 的 `_TEXT_GENERATION_MODELS`（或对应任务表）加一行 `"XxxForCausalLM": ("xxx", "XxxForCausalLM")`；
3. 实现标准 `__init__(vllm_config, prefix)` / `embed_input_ids` / `forward(input_ids, positions, ...)` / `compute_logits`；
4. 写 `load_weights`：处理融合矩阵、量化 scale、PP missing、特殊命名；
5. 如果需要 TP 切分，用 `ColumnParallelLinear`/`RowParallelLinear`/`QKVParallelLinear`/`MergedColumnParallelLinear`/`VocabParallelEmbedding`；
6. 按需加 `Supports*`（LoRA/PP/多模态/MoE/Hybrid）；
7. 多模态模型还要在 `MULTIMODAL_REGISTRY` 注册 processor、定义占位符与编码器。

模型类本身**不含调度、KV cache 管理、采样、注意力后端选择**——这些全在 worker/attention/sample 层。模型只负责"给定输入张量和位置，输出隐状态"，这是它能被统一引擎驱动 290+ 种架构的根本原因。

---

## 小结

- **`ModelRegistry` 用架构字符串映射到 "(模块, 类)"，延迟导入 + 子进程 inspect + 缓存**，把 290+ 模型的加载成本摊到真正用到时；
- **`VllmModel` 协议极薄**（`__init__(vllm_config)` + `embed_input_ids` + `forward`），生成模型加 `compute_logits`，池化模型加 `pooler`；`Supports*` 协议按需开启多模态/LoRA/PP/MoE/Hybrid 等能力；
- **标准结构是 Embedding → N×(Attention/MLP block with fused add-norm) → Norm → LM Head**，PP 用 `intermediate_tensors` 串接，层列表用 `make_layers` 按 rank 切；
- **权重加载靠参数级 `weight_loader` + 模块级 `load_weights` + `AutoWeightsLoader` 三层钩子**，处理融合矩阵、TP 切片、量化 scale、PP missing、专家映射；
- **多模态、MoE、Hybrid/SSM 都是在同一契约上的扩展**：视觉嵌入在 `embed_input_ids` 合并，MoE 用 `FusedMoE`，SSM 层与 Attention 层共用 `AttentionLayerBase` 接口；
- 模型文件因此退化为"用积木搭网络 + 写权重名字映射"，引擎、worker、注意力后端、采样器对所有模型一视同仁。

下一篇 [07 编译管线](../performance/07_compilation.md) 将进入 `vllm/compilation/`，看 `torch.compile`、piecewise backend、CUDA graph 捕获、编译器 pass 如何把这里的模型 forward 变成高性能可重放的计算图。
