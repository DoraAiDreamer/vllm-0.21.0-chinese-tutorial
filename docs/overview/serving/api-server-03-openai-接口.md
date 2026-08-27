# API Server 之三：OpenAI 接口实现

> 相关源码：`vllm/entrypoints/openai/`

本篇看 `entrypoints/openai/` 下各 OpenAI 兼容端点的实现模式。每个子领域基本是三件套：`protocol.py`（Pydantic 模型）、`serving.py`（业务类，调 `engine_client`）、`api_router.py`（FastAPI 路由）。公共基类在 `openai/serving.py`（`OpenAIServing`）。

---

## 1. 通用模式

### 1.1 三段式文件

以 chat completion 为例：

```
chat_completion/
├── protocol.py    # ChatCompletionRequest / ChatCompletionResponse / Chunk / Choice ...
├── serving.py     # OpenAIServingChat：渲染请求、调 engine、把 RequestOutput 转成 OpenAI 响应
├── api_router.py  # APIRouter，定义 POST /v1/chat/completions
├── batch_serving.py       # 批处理服务（OpenAIServingChatBatch）
└── stream_harmony.py      # 与 Responses API 复用的流式事件协调
```

`api_router.attach_router(app)` 把路由挂到 FastAPI；端点函数从 `request.app.state` 拿到 serving 单例。

### 1.2 OpenAIServing 基类

[openai/serving.py:136](../../vllm/entrypoints/openai/serving.py#L136) 提供所有 serving 类的公共能力：

- 持有 `engine_client`、`model_config`、`models`（`OpenAIServingModels`）、`request_logger`、渲染器；
- `_check_model(model_name)`：校验请求的 `model` 字段在已加载模型/LoRA 列表中；
- `create_error_response`/`create_streaming_error_response`：统一错误格式；
- `_raise_if_error`：把引擎返回的 `finish_reason="error"` 转成异常；
- `beam_search(...)`：束搜索的生成器封装；
- `_maybe_get_adapters(request)`：解析请求里的 LoRA（含多模态默认 LoRA）；
- `_validate_chat_template`、`_prepare_extra_chat_template_kwargs`、`_get_decoded_token`、`_parse_tool_calls_from_content` 等工具。

### 1.3 端点函数骨架

以 chat 为例（`api_router.py`）：

```python
@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    serving = chat(raw_request)              # 从 app.state 取 serving 实例
    if (error := await serving._check_model(request.model)):
        return error
    return await serving.create_chat_completion(request, raw_request)
```

`serving.create_chat_completion` 根据 `stream` 选择流式生成器或一次性生成器，返回 FastAPI 可识别的 `StreamingResponse`（SSE）或 Pydantic 响应对象。

---

## 2. Chat Completions

`serving.py:82` 的 `OpenAIServingChat` 是最核心、最复杂的端点。

### 2.1 请求处理流程（[_create_chat_completion](../../vllm/entrypoints/openai/chat_completion/serving.py#L241)）

1. **渲染 chat template**：`render_chat_request(request)` 用渲染器把 messages/tools/chat_template_kwargs 转成 `EngineInput`（token/embedding + 多模态），返回 `(conversation, engine_inputs)`。这一步在 API 进程完成，支持请求自带模板、内置模板、HF tokenizer chat template；
2. **request_id**：`chatcmpl-{base_request_id}`，挂到 `raw_request.state.request_metadata`；
3. **LoRA 解析**：`_maybe_get_adapters`；
4. **DP rank**：从 header 读取路由注入的 data-parallel rank；
5. **构造 SamplingParams**：`request.to_sampling_params(max_tokens, defaults)`，或 beam search 的 `BeamSearchParams`；自动算 `max_tokens`（`max_model_len - prompt_len`，受 `truncate_prompt_tokens` 影响）；
6. **reasoning 处理**：若请求关闭 reasoning、或用了带 grammar 的 tool parser、或 `ReasoningParser` 判定 prompt 已包含结束标记，设置 `reasoning_ended`；
7. **调引擎**：`engine_client.generate(engine_input, sampling_params, sub_request_id, lora_request, trace_headers, priority, data_parallel_rank, reasoning_ended, ...)`（[L347](../../vllm/entrypoints/openai/chat_completion/serving.py#L347)），得到 `RequestOutput` 异步生成器；多 prompt 时建立多个子生成器；
8. **返回**：流式走 `chat_completion_stream_generator`，非流式走 `chat_completion_full_generator`。

### 2.2 流式生成器

[chat_completion_stream_generator](../../vllm/entrypoints/openai/chat_completion/serving.py#L397) 把 `RequestOutput` 转成 OpenAI SSE 事件：

- 首包发 `role` chunk；
- 增量输出 content（delta）或 reasoning_content；
- **tool calls 增量拼装**：用 `_should_stream_with_auto_tool_parsing`、`_create_remaining_args_delta` 处理流式工具调用参数（JSON 可能跨 chunk）；
- logprobs：`_create_chat_logprobs`/`_get_top_logprobs`；
- `usage` 在最后一包发送（含 prompt/token 统计）；
- 结束时发 `finish_reason`（stop/length/tool_calls/content_filter 等）和 `[DONE]`；
- `stream_options.include_usage` 控制是否返回 usage。

### 2.3 非流式生成器

[chat_completion_full_generator](../../vllm/entrypoints/openai/chat_completion/serving.py#L1003) 收集所有 `RequestOutput`，拼接完整文本/reasoning、聚合 logprobs、构造 `ChatCompletionResponse`。

### 2.4 Batch 与 Harmony

- `batch_serving.py` 的 `OpenAIServingChatBatch` 处理批量请求（离线/批处理服务）；
- `stream_harmony.py` 让 Chat 与 Responses API 共享一套流式事件协调逻辑。

---

## 3. Completions

`completion/serving.py:51` 的 `OpenAIServingCompletion` 是更老的 text completion 端点：

- `render_completion_request` 把 prompt（字符串/字符串数组/token 数组）渲染成 `EngineInput`；
- `_create_completion` 构造 `SamplingParams`，调 `engine_client.generate`；
- 流式：`completion_stream_generator`（[L276](../../vllm/entrypoints/openai/completion/serving.py#L276)）产出 `CompletionResponseStreamChoice`；
- 非流式：`request_output_to_completion_response`（[L473](../../vllm/entrypoints/openai/completion/serving.py#L473)）；
- logprobs 由 `_create_completion_logprobs` 处理。

端点：`POST /v1/completions`。

---

## 4. Models

`models/serving.py` 提供：

- **`OpenAIModelRegistry`**（[L31](../../vllm/entrypoints/openai/models/serving.py#L31)）：基础模型名集合与 base model 判断；
- **`OpenAIServingModels`**（[L76](../../vllm/entrypoints/openai/models/serving.py#L76)）：
  - `init_static_loras`：启动时加载 `--lora-modules`；
  - `show_available_models`：`GET /v1/models` 返回 base model + 已加载 LoRA（`ModelList`）；
  - `load_lora_adapter`/`unload_lora_adapter`：运行时动态加载/卸载 LoRA（`POST /v1/load_lora_adapters` 等）；
  - `resolve_lora`：把请求的 `model` 名解析成 `LoRARequest`。

LoRA 适配器通过 engine client 的 `add_lora/remove_lora` 生效。

---

## 5. Responses API

`responses/` 实现 OpenAI 较新的 `/v1/responses` 接口（支持多轮内置工具、状态化会话）。

- `protocol.py` 定义 `ResponsesRequest`、`Response`、事件流（`response.created`、`response.output_text.delta`、`response.function_call_arguments.delta`、`response.completed` 等）；
- `serving.py:152` 的 `OpenAIServingResponses`：
  - `create_responses` → `_create_responses`（[L331](../../vllm/entrypoints/openai/responses/serving.py#L331)）；
  - `_render_next_turn`、`_generate_with_builtin_tools`（[L642](../../vllm/entrypoints/openai/responses/serving.py#L642)）：支持内置 web search/file search/code interpreter 等工具的多轮循环；
  - `responses_full_generator`（[L761](../../vllm/entrypoints/openai/responses/serving.py#L761)）把输出转成 Responses 事件；
  - `harmony.py`/`context.py` 维护会话上下文与流式事件；
- 路由（`api_router.py`）：
  - `POST /v1/responses`（创建，支持流式 SSE）；
  - `GET /v1/responses/{response_id}`（查询）；
  - `POST /v1/responses/{response_id}/cancel`（取消）；
- 流式输出在 `api_router._convert_stream_to_sse_events` 中把内部事件转成 SSE。

它与 Chat Completions 共享渲染器和大量逻辑，但输出格式是 OpenAI Responses 事件而非 `chat.completion.chunk`。

---

## 6. Realtime API（语音/多模态 WebSocket）

`realtime/` 实现 `GET /v1/realtime`（WebSocket 升级）：

- `api_router.py:28` 的 `realtime_endpoint(websocket)`：建立 `RealtimeConnection`；
- `connection.py:32` 的 `RealtimeConnection`：管理 WebSocket 会话生命周期——接收客户端事件（`conversation.item.create`、`response.create`、`session.update` 等）、驱动引擎、通过 WebSocket 推送服务端事件（`response.created`、`response.audio.delta`、`rate_limits.updated`）；
- `serving.py` 处理会话配置、音频缓冲、VAD、与 engine 的流式输入/输出对接；
- `protocol.py` 定义 Realtime 事件；
- `metrics.py` 提供 WebSocket 指标中间件。

它与普通 HTTP 端点最大的不同是**双向流式**：输入可能边到边生成（`StreamingInput`，见第 2 篇），输出也以音频/文本 delta 实时推送。

---

## 7. 其他生成类端点

### 7.1 Generative Scoring

`generative_scoring/serving.py:145` 的 `OpenAIServingGenerativeScoring`：用生成模型做评分/重排（生成式 reranker），`POST /v1/generative_scoring`。它把 query/document 渲染成 prompt，批量生成并解析打分，返回 `GenerativeScoringResponse`。

### 7.2 Generate（非 OpenAI 兼容的低层接口）

`generate/` 提供 `/generate`（vLLM 早期的 HTTP 接口，非 OpenAI 风格），由 `register_generate_api_routers` 挂载；`factories.py` 构造请求对象。

### 7.3 Speech-to-Text

`speech_to_text/` 提供 ASR 相关端点（转录/翻译），用 `transcription` 任务模型（如 Whisper 类）。

### 7.4 Tokenization

`engine/serving.py` 的 `OpenAIServingTokenization`（在 `init_app_state` 里创建）提供 `/tokenize`、`detokenize`、`/v1/chat_template` 等工具端点，直接用渲染器/tokenizer。

### 7.5 Render

`serve/render/` 提供"只渲染不推理"的端点（`/v1/chat/render` 等），用于调试 chat template、工具解析、token 计数，不调用 GPU 引擎。

---

## 8. 公共的 Serving 子模块

`openai/` 下还有一些跨端点复用的模块：

- `chat_utils.py`：消息处理、多模态消息解析、工具调用相关；
- `parser/`：`responses_parser.py`、`harmony_utils.py`——把模型输出解析成 Responses API 事件；
- `realtime/`、`generative_scoring/` 各自的 protocol；
- `models/` 见第 4 节；
- `server_utils.py`：`AuthenticationMiddleware`、`XRequestIdMiddleware`、`RequestLogger`、SageMaker 引导等（第 5 篇详述）；
- `orca_metrics.py`：Prometheus 指标。

---

## 9. 响应格式与流式约定

- **流式 SSE**：`Content-Type: text/event-stream`，每行 `data: <json>\n\n`，结束发送 `data: [DONE]\n\n`（chat/completions）或 `response.completed` 事件（responses）；
- **非流式**：直接返回 Pydantic JSON；
- **错误**：`ErrorResponse`（`{"error": {"message", "type", "code", "param"}}`），由 `build_app` 注册的异常处理器统一生成；
- **usage**：`prompt_tokens`/`completion_tokens`/`total_tokens`，流式在末包发送；
- **logprobs**：按请求参数返回 token 级对数概率与 top-k；
- **request id**：透传到 `RequestResponseMetadata`，并可通过 `X-Request-Id` 头关联。

---

## 小结

1. **每个 OpenAI 端点都是 protocol + serving + api_router 三件套**，公共能力在 `OpenAIServing` 基类；
2. **Chat Completions 最复杂**：chat template 渲染、SamplingParams 构造、reasoning/tool calls 增量流式、logprobs、beam search、多 prompt 子请求；
3. **Completions** 是更简单的文本版本；
4. **Models** 管理 base model 与运行时 LoRA；
5. **Responses** 是新的多轮工具 API，支持内置工具循环与独特事件流；
6. **Realtime** 是 WebSocket 双向流式，用于语音/多模态实时交互；
7. 还有 generative scoring、generate、speech-to-text、tokenization、render 等辅助/专用端点；
8. 所有 serving 类最终都通过 `engine_client.generate/encode` 进入第 2 篇的请求流水线。

下一篇 [API Server 之四：批处理与扩展服务](./api-server-04-批处理与扩展.md) 将介绍 `run_batch`（OpenAI 批量文件处理）以及 `vllm/entrypoints/serve/` 下的 disagg（PD 分离）、elastic EP、LoRA、sleep、RLHF、profile、tokenize 等管理 API。
