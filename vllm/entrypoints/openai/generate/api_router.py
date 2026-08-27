# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from fastapi import FastAPI

if TYPE_CHECKING:
    from argparse import Namespace

    from starlette.datastructures import State

    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.logger import RequestLogger
    from vllm.tasks import SupportedTask
else:
    RequestLogger = object


def register_generate_api_routers(app: FastAPI):
    from vllm.entrypoints.openai.chat_completion.api_router import (
        attach_router as register_chat_api_router,
    )

    register_chat_api_router(app)

    from vllm.entrypoints.openai.responses.api_router import (
        attach_router as register_responses_api_router,
    )

    register_responses_api_router(app)

    from vllm.entrypoints.openai.completion.api_router import (
        attach_router as register_completion_api_router,
    )

    register_completion_api_router(app)

    from vllm.entrypoints.anthropic.api_router import (
        attach_router as register_anthropic_api_router,
    )

    register_anthropic_api_router(app)


async def init_generate_state(
    engine_client: "EngineClient",
    state: "State",
    args: "Namespace",
    request_logger: RequestLogger | None,
    supported_tasks: tuple["SupportedTask", ...],
):
    """
    放在 app.state 是为了拿到一个应用级、生命周期受控、无 import 副作用、可被路由依赖注入访问的单例：
    serving 服务本身是跨请求共享的无状态/线程安全资源，在启动时 init_app_state 建好挂到 state，
    每个 HTTP 请求再通过 request.app.state.xxx 取用，既保证全局唯一，又避免了用裸全局变量带来的初始化时机和测试隔离问题。

    初始化"生成类(generate)"任务的全部 OpenAI/Anthropic 服务对象。

    在服务启动 lifespan 阶段调用，把各 serving 服务挂载到 FastAPI 的 app.state
    上，供 HTTP 路由通过依赖辅助函数（如 chat(request)）取用。仅在
    supported_tasks 含 "generate" 时创建对应服务，否则置为 None。

    挂载到 state 上的对象包括：
    openai_serving_responses / openai_serving_chat / openai_serving_chat_batch /
    openai_serving_completion / anthropic_serving_messages / serving_tokens。
    """
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
    from vllm.entrypoints.chat_utils import load_chat_template
    from vllm.entrypoints.mcp.tool_server import (
        DemoToolServer,
        MCPToolServer,
        ToolServer,
    )
    from vllm.entrypoints.openai.chat_completion.batch_serving import (
        OpenAIServingChatBatch,
    )
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.fingerprint import set_default_fingerprint_mode
    from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
    from vllm.entrypoints.serve.disagg.serving import ServingTokens

    # Applied before any serving class is constructed so that each one picks
    # up the chosen mode on its first cache miss.
    set_default_fingerprint_mode(
        getattr(args, "fingerprint_mode", "full"),
        getattr(args, "fingerprint_value", None),
    )

    if args.tool_server == "demo":
        tool_server: ToolServer | None = DemoToolServer()
        assert isinstance(tool_server, DemoToolServer)
        await tool_server.init_and_validate()
    elif args.tool_server:
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(args.tool_server)
    else:
        tool_server = None
    resolved_chat_template = load_chat_template(args.chat_template)

    # Render endpoints are always backed by OpenAIServingRender so that
    # /v1/chat/completions/render and /v1/completions/render work on both
    # generate-mode and render-only servers. Created in init_app_state.

    # /v1/responses 服务：OpenAI Responses API（多轮内置工具、事件流），不支持则为 None
    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            tool_server=tool_server,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
        )
        if "generate" in supported_tasks
        else None
    )
    _chat_kwargs = dict(
        engine_client=engine_client,
        models=state.openai_serving_models,
        response_role=args.response_role,
        openai_serving_render=state.openai_serving_render,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        default_chat_template_kwargs=args.default_chat_template_kwargs,
        trust_request_chat_template=args.trust_request_chat_template,
        return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        enable_auto_tools=args.enable_auto_tool_choice,
        exclude_tools_when_tool_choice_none=args.exclude_tools_when_tool_choice_none,
        tool_parser=args.tool_call_parser,
        reasoning_parser=args.structured_outputs_config.reasoning_parser,
        enable_prompt_tokens_details=args.enable_prompt_tokens_details,
        enable_force_include_usage=args.enable_force_include_usage,
        enable_log_outputs=args.enable_log_outputs,
        enable_log_deltas=args.enable_log_deltas,
    )
    # /v1/chat/completions 服务：渲染 chat template、流式/非流式 chat 补全
    state.openai_serving_chat = (
        OpenAIServingChat(**_chat_kwargs) if "generate" in supported_tasks else None
    )
    # 批量 chat 补全服务：一次请求处理多个 prompt（离线/批处理端点）
    state.openai_serving_chat_batch = (
        OpenAIServingChatBatch(**_chat_kwargs)
        if "generate" in supported_tasks
        else None
    )
    if state.openai_serving_chat is not None:
        state.openai_serving_chat.warmup()
    # /v1/completions 服务：文本补全（不做 chat 对话封装），不支持则为 None
    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )
    # Anthropic Messages API 服务：/v1/messages 兼容端点，不支持则为 None
    state.anthropic_serving_messages = (
        AnthropicServingMessages(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
        )
        if "generate" in supported_tasks
        else None
    )
    # PD 分离的 /inference/v1/generate 服务：prefill/decode 实例间 KV 传输协调
    state.serving_tokens = (
        ServingTokens(
            engine_client,
            state.openai_serving_models,
            state.openai_serving_render,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_log_outputs=args.enable_log_outputs,
            force_no_detokenize=args.tokens_only,
        )
        if "generate" in supported_tasks
        else None
    )
