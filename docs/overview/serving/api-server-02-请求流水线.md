# API Server 之二：请求处理流水线

> 相关源码：`vllm/v1/engine/async_llm.py`、`input_processor.py`、`output_processor.py`、`detokenizer.py`、`request.py`、`v1/request.py`

本篇追踪一个生成请求从进入 `AsyncLLM` 到产生流式 `RequestOutput` 的完整路径。核心是两个处理器（Input/Output）和两个后台循环（output_handler、EngineCore 的 busy loop）。

---

## 1. 全链路鸟瞰

```
API 端点 (serving.py)
   │  request_id, prompt/EngineInput, SamplingParams
   ▼
AsyncLLM.generate()                         async_llm.py:524
   ├─ add_request()                         async_llm.py:280
   │     ├─ InputProcessor.process_inputs()        构造 EngineCoreRequest
   │     ├─ assign_request_id()                   内部/外部请求 id
   │     ├─ RequestOutputCollector 建立           每请求的异步队列
   │     └─ _add_request() → engine_core.add_request_async()  ── ZMQ ──┐
   │                                                                   │
   │  (同时) output_handler 后台 Task              async_llm.py:637      │
   │     loop: engine_core.get_output_async()  ◀────────────────────────┘
   │           OutputProcessor.process_outputs()
   │           collector.put(RequestOutput)
   │                                                                    
   └─ async for out in collector.get():      拉取并 yield 给端点
         yield RequestOutput
```

EngineCore 侧（独立进程，第 04 篇）：收到 `EngineCoreRequest` 后加入 Scheduler，`step()` 调度执行并产出 `EngineCoreOutputs` 经 ZMQ 回传。

整个流水线的关键设计是**全异步、背压、每请求独立队列**：API 端点只需 `async for` 输出，HTTP 断开时自动 abort。

---

## 2. 数据结构

### 2.1 Request（`v1/request.py:59`）

引擎内部的请求对象（区别于 `EngineCoreRequest` 这个线上消息）：
- 持有 `request_id`、`prompt_token_ids`/`prompt_embeds`、`params`、`sampling_params`、`lora_request`、`arrival_time`、`priority`；
- 多模态 `encoder_inputs`；
- 输出状态：`output_token_ids`、`finished`、`events`、`prefill_stats`；
- `from_engine_core_request` 从消息反序列化；
- `append_output_token_ids`、`num_tokens`/`num_output_tokens`、`record_event`/`take_events`、`take_prefill_stats`；
- `__lt__` 按 `arrival_time` 比较（用于优先队列）。

### 2.2 EngineCoreRequest（`vllm/v1/engine/__init__.py:81`）

跨进程消息，已在第 08 篇相关文档中注解：prompt token/embedding、采样或池化参数、LoRA、mm_features、DP rank、client_index、wave/priority、external_req_id、abort_immediately 等。

### 2.3 RequestOutputCollector（`output_processor.py:48`）

每个请求一个的异步桥接器：
- `put(output)` 由 output_handler 调用；
- `get()` / `get_nowait()` 由 `generate()` 循环调用；
- `close()` 在结束/取消时清理；
- 支持 `RequestOutputKind`（按 `params.output_kind`，用于 `n>1` 的 fan-out/fan-in）。

### 2.4 RequestState / OutputProcessorOutput

`RequestState`（[output_processor.py:132](../../vllm/v1/engine/output_processor.py#L132)）持有一个请求在输出处理阶段的全部状态：detokenizer、logprobs 处理器、累计 token、prompt logprobs、是否在 prefill、`RequestOutput` 模板、tracing/统计。`OutputProcessorOutput` 是一步处理的返回集合（待推送的输出、要 abort 的请求等）。

---

## 3. 入口：AsyncLLM.generate

[generate](../../vllm/v1/engine/async_llm.py#L524) 是一个 async generator：

1. 调 `add_request(...)` 拿到 `RequestOutputCollector`；
2. 循环 `out = q.get_nowait() or await q.get()`：先非阻塞排空队列（高负载下减少 task 切换），再等待；
3. `finished = out.finished`，只要不是哨兵 `STREAM_FINISHED` 就 `yield out`；
4. **取消/断开**：捕获 `asyncio.CancelledError`/`GeneratorExit`，调 `abort(request_id, internal=True)` 通知引擎释放资源；
5. **异常分类**：
   - `EngineDeadError`：引擎已死，不再 abort；
   - `ValueError`：请求校验错误（bad request）；
   - `InputStreamError`：流式输入错误；
   - 其他 `Exception`：包装成 `EngineGenerateError`；
6. `finally: q.close()`。

`generate` 的 docstring 明确了四步：建 AsyncStream → 处理输入 → 加入 Detokenizer → 加入 EngineCore。注意 detokenizer 实际由 OutputProcessor 在收到首个输出时创建。

### n>1 的 fan-out

若 `sampling_params.n > 1`，`add_request` 会创建一个 `ParentRequest`，复制出 n 个 `child_request`（各自 request_id），全部加入引擎，但共享同一个 collector（[async_llm.py:385-398](../../vllm/v1/engine/async_llm.py#L385)）。多个子请求的输出汇入同一队列返回给调用方。

---

## 4. 输入处理：InputProcessor

[InputProcessor](../../vllm/v1/engine/input_processor.py#L36) 负责把"用户/上层传来的 prompt"变成可发给引擎的 `EngineCoreRequest`。

### 4.1 构造与依赖

- `vllm_config`、`renderer`（chat template/tokenizer）、`input_preprocessor`；
- 通过 `tokenizer`/`get_tokenizer` 暴露分词器；
- 校验逻辑：`_validate_params`、`_validate_lora`、`_validate_prompt_len`、`_validate_model_input`。

### 4.2 process_inputs 主流程

[input_processor.py:234](../../vllm/v1/engine/input_processor.py#L234)：

1. **校验参数与 LoRA**：`_validate_params(params, supported_tasks)`、`_validate_lora(lora_request)`；
2. **DP rank 校验**：检查 `data_parallel_rank` 在 `[0, num_ranks)` 范围；
3. **预处理 prompt**：
   - 现代路径：prompt 已是 `EngineInput` TypedDict（由 `Renderer.render_cmpl()/render_chat()` 产出，含 `type`、`prompt_token_ids`/`prompt_embeds`、多模态字段）；
   - 兼容旧路径：原始 prompt（字符串/列表/tokens）走 `input_preprocessor.preprocess(...)`；
4. **平台校验**：`current_platform.validate_request(processed_inputs, params)`；
5. **拆分 encoder/decoder 输入**（多模态 ASR/encoder-decoder）：`split_enc_dec_inputs`；
6. **取出 prompt_token_ids / prompt_embeds / is_token_ids**；
7. **处理采样参数**：
   - 克隆 `SamplingParams`；
   - 未设 `max_tokens` 时填 `max_model_len - prompt_len`；
   - `update_from_generation_config`（应用模型 generation config，如温度/top_p/eos/repetition_penalty）；
   - `update_from_tokenizer`（用 tokenizer 的 eos 等补全）；
8. **多模态特征组装**：当 `decoder_inputs["type"] == "multimodal"` 时，按占位位置排序，把 `(data, modality, identifier, mm_position, mm_hash)` 打包成 `MultiModalFeatureSpec` 列表。`_get_mm_identifier` 结合 hash 与 LoRA 生成缓存标识；
9. **构造并返回 `EngineCoreRequest`**：填充 request_id、prompt、mm_features、params、arrival_time、lora、DP rank、client_index、wave、priority、external_req_id、resumable 等。

### 4.3 多模态缓存注入

`inject_into_mm_cache`（[L175](../../vllm/v1/engine/input_processor.py#L175)）支持把已预处理的多模态数据直接注入 encoder cache（配合多处理器/共享内存）。`assign_request_id`（[L215](../../vllm/v1/engine/input_processor.py#L215)）把外部 request_id 拷贝到 `external_req_id`，内部用自增/唯一 id，用于 `abort(req_id, internal=False)` 这样的外部 API。

### 4.4 流式输入

`_add_streaming_input_request`（[async_llm.py:417](../../vllm/v1/engine/async_llm.py#L417)）支持 prompt 是一个 `AsyncGenerator[StreamingInput, None]`（如实时语音/增量文本），边到边发给引擎；`_validate_streaming_input_sampling_params` 限制其与某些特性互斥。

---

## 5. 输出处理：OutputProcessor 与 output_handler

### 5.1 output_handler 后台任务

[_run_output_handler](../../vllm/v1/engine/async_llm.py#L637) 是一个 `asyncio.Task`：
- 循环 `outputs = await self.engine_core.get_output_async()`；
- 调 `self.output_processor.process_outputs(outputs, ...)`；
- 把每个 `RequestOutput` 放入对应请求的 `RequestOutputCollector`；
- 处理 abort、错误传播（`propagate_error`）。

它在 `__init__` 中若检测到运行中的 event loop 就立即启动；否则在第一次 `add_request` 时启动（[L373](../../vllm/v1/engine/async_llm.py#L373)），以便在 OpenAI server 里优雅处理启动失败。

### 5.2 process_outputs

[process_outputs](../../vllm/v1/engine/output_processor.py#L597) 是唯一应遍历整个批次的函数（注释强调最小化 Python 循环）。对每个 `EngineCoreOutput`：

1. 取 `RequestState`（已 abort 的请求直接忽略）；
2. `_update_stats_from_output`：更新迭代统计、tracing；
3. prefill 阶段：若有 `prefill_stats` 记录 `num_cached_tokens`，标记 prefill 结束；
4. **生成模型**（`pooling_output is None`）：
   - 调 detokenizer 把 `new_token_ids` 转文本并做 stop 检查；
   - 处理 logprobs（采样 logprobs、prompt logprobs 累积）；
   - `req_state.make_request_output(...)` 组装 `RequestOutput`；
   - 若请求结束（finish_reason 非空），`_finish_request` 做收尾；
5. **池化模型**：用 `pooling_output` 构造 `PoolingRequestOutput`；
6. 收集 `routed_experts`、`kv_transfer_params`、事件、NaN 计数等；
7. 推送到 collector（有队列时）或返回列表（LLMEngine 同步路径）。

### 5.3 请求状态管理

- `add_request`（[L533](../../vllm/v1/engine/output_processor.py#L533)）：为新请求创建 `RequestState`（含 detokenizer 工厂、logprobs 处理器），加入 `request_states`；
- `abort_requests(request_ids, internal)`（[L471](../../vllm/v1/engine/output_processor.py#L471)）：标记中止、生成 abort 输出、从状态表移除；
- `_finish_request`（[L714](../../vllm/v1/engine/output_processor.py#L714)）：记录完成统计、清理；
- `propagate_error`：把异常发给所有未完成请求的 collector。

---

## 6. 增量解码：IncrementalDetokenizer

[detokenizer.py](../../vllm/v1/engine/detokenizer.py) 把每步新增的 token id 高效地增量转成文本，并处理停止串。

### 6.1 类层次

- `IncrementalDetokenizer`（接口，[L30](../../vllm/v1/engine/detokenizer.py#L30)）：`update`、`get_next_output_text`、`output_token_ids`、`num_output_tokens`、`from_new_request`；
- `BaseIncrementalDetokenizer`（[L68](../../vllm/v1/engine/detokenizer.py#L68)）：通用逻辑，含 prefix 偏移、stop string 检查、"下一段输出文本"计算；
- `FastIncrementalDetokenizer`（[L167](../../vllm/v1/engine/detokenizer.py#L167)）：用 `PreTrainedTokenizerFast`（HF Rust tokenizer）的 `decode` 批次解码，性能更好；
- `SlowIncrementalDetokenizer`（[L250](../../vllm/v1/engine/detokenizer.py#L250)）：慢速回退，兼容非 fast tokenizer。

### 6.2 工作流

1. `update(new_token_ids, stop_terminated)`（[L95](../../vllm/v1/engine/detokenizer.py#L95)）：累积新 token，检测停止串（`check_stop_strings`），必要时回滚被停止串"吞掉"的 token；
2. `get_next_output_text(finished, delta=True/False)`：
   - `delta=True`（流式）：返回自上次调用以来的**新增文本**；
   - `delta=False`（非流式）：返回完整解码文本；
3. 处理 Unicode 边界（一个字符可能跨多个 token，需要等字符完整才输出）；
4. `finished` 时刷新所有剩余文本。

detokenizer 与 `stream_interval`（`SamplingParams`/scheduler 配置）配合：只有到了间隔步才真正产出新文本，减少 Python 开销。

---

## 7. 流式 vs 非流式

- **流式**（默认 `stream=True`）：每产生 `stream_interval` 个 token（默认 1），OutputProcessor 就生成一个 `RequestOutput`，output_handler 立即推送到 collector，端点以 SSE（`data: {...}\n\n`）逐块返回 `ChatCompletionChunk`。
- **非流式**（`stream=False`）：detokenizer 的 `delta=False`，但引擎仍逐 token 输出；端点在 `async for` 里只保留最终结果（或在 collector 层聚合），最后返回完整 `ChatCompletion`。
- `RequestOutputKind`/`output_kind` 决定 collector 行为（如 `FINAL_ONLY`）。

---

## 8. 错误、取消与清理

- **客户端断开**：FastAPI 取消请求任务 → `generate` 捕获 `CancelledError` → `abort(internal=True)` → 引擎在下一步释放该请求的 KV block；
- **校验错误**：在 `process_inputs` 阶段抛出 `ValueError`，请求从未进入引擎；
- **引擎死亡**：`EngineDeadError` 向上传播，API 返回 500；
- **输出阶段异常**：`propagate_error` 下发到所有 collector，`generate` 转成 `EngineGenerateError`；
- **abort API**：端点的"删除请求"调 `engine_client.abort(request_id)`，区分 internal（引擎内部清理）与 external（用户调用，用 external_req_id）。

---

## 9. 池化模型路径（encode）

[AsyncLLM.encode](../../vllm/v1/engine/async_llm.py#L801) 处理 embedding/score/classify：
- 用 `PoolingParams` 而非 `SamplingParams`；
- 同样经 `InputProcessor`/`add_request`，但请求进入引擎后由 pooling 模型输出 `pooling_output`；
- OutputProcessor 走 `_new_pooling_output` 分支，产出 `PoolingRequestOutput`；
- 不做 detokenize、不流式（通常一步返回）。

---

## 小结

1. **`AsyncLLM.generate` 是异步生成器**，内部 `add_request` 入队、`RequestOutputCollector` 出队，客户端断开自动 abort；
2. **InputProcessor** 把 `EngineInput`/prompt 校验、补全采样参数、组装多模态特征，产出跨进程的 `EngineCoreRequest`；
3. **OutputProcessor** 是唯一遍历批次输出的地方，负责 detokenize、logprobs、stop、统计、组装 `RequestOutput` 并推送到每请求队列；
4. **IncrementalDetokenizer** 增量解码并处理 stop string/Unicode 边界，支持 fast/slow 两种 tokenizer 与流式 delta；
5. **output_handler 后台 Task** 在 engine 与生成器之间做缓冲与分发，`n>1` 通过父子请求 fan-out/fan-in；
6. 池化模型走 `encode` 路径，输出 `PoolingRequestOutput`。

下一篇 [API Server 之三：OpenAI 接口实现](./api-server-03-openai-接口.md) 将看 `entrypoints/openai/` 下的 serving 层如何把这些 `RequestOutput` 包装成 OpenAI 兼容的 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、Responses、Realtime 等接口。
