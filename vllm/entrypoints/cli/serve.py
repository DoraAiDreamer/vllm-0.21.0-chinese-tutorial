# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import signal
import time

import uvloop

import vllm
import vllm.envs as envs
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.api_server import run_server, setup_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_tcp_uri
from vllm.v1.engine.utils import CoreEngineProcManager, launch_core_engines
from vllm.v1.executor import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus
from vllm.v1.utils import APIServerProcessManager, wait_for_completion_or_failure

logger = init_logger(__name__)

DESCRIPTION = """Launch a local OpenAI-compatible API server to serve LLM
completions via HTTP. Defaults to Qwen/Qwen3-0.6B if no model is specified.

Search by using: `--help=<ConfigGroup>` to explore options by section (e.g.,
--help=ModelConfig, --help=Frontend)
  Use `--help=all` to show all available flags at once.
"""


class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI.

    `vllm serve` 子命令：启动 OpenAI 兼容 API 服务器（含 headless、
    单/多 API server、gRPC 等模式），是推理服务的主入口。
    """

    name = "serve"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """子命令执行入口：解析部署模式并分发到对应启动路径。

        顺序：处理位置参数 model → gRPC 模式 → headless 模式 →
        根据数据并行负载均衡模式确定 api_server_count →
        分发到 run_headless(0 个) / run_multi_api_server(>1 个) /
        run_server(单个，uvloop 内联运行)。
        """
        # If model is specified in CLI (as positional arg), it takes precedence
        if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

        if getattr(args, "grpc", False):
            from vllm.entrypoints.grpc_server import serve_grpc

            uvloop.run(serve_grpc(args))
            return

        if args.headless:
            if args.api_server_count is not None and args.api_server_count > 0:
                raise ValueError(
                    f"--api-server-count={args.api_server_count} cannot be "
                    "used with --headless (no API servers are started in "
                    "headless mode)."
                )
            # Default to 0 in headless mode (no API servers)
            args.api_server_count = 0

        # Detect LB mode for defaulting api_server_count.
        # External LB: --data-parallel-external-lb or --data-parallel-rank
        # Hybrid LB: --data-parallel-hybrid-lb or --data-parallel-start-rank
        is_external_lb = (
            args.data_parallel_external_lb or args.data_parallel_rank is not None
        )
        is_hybrid_lb = (
            args.data_parallel_hybrid_lb or args.data_parallel_start_rank is not None
        )

        if is_external_lb and is_hybrid_lb:
            raise ValueError(
                "Cannot use both external and hybrid data parallel load "
                "balancing modes. External LB is enabled via "
                "--data-parallel-external-lb or --data-parallel-rank. "
                "Hybrid LB is enabled via --data-parallel-hybrid-lb or "
                "--data-parallel-start-rank. Use one mode or the other."
            )

        # Default api_server_count if not explicitly set.
        # - External LB: Leave as 1 (external LB handles distribution)
        # - Hybrid LB: Use local DP size (internal LB for local ranks only)
        # - Internal LB: Use full DP size
        if args.api_server_count is None:
            if is_external_lb:
                args.api_server_count = 1
            elif is_hybrid_lb:
                args.api_server_count = args.data_parallel_size_local or 1
                if args.api_server_count > 1:
                    logger.info(
                        "Defaulting api_server_count to data_parallel_size_local "
                        "(%d) for hybrid LB mode.",
                        args.api_server_count,
                    )
            else:
                args.api_server_count = args.data_parallel_size
                if args.api_server_count > 1:
                    logger.info(
                        "Defaulting api_server_count to data_parallel_size (%d).",
                        args.api_server_count,
                    )

        # Elastic EP currently only supports running with at most one API server.
        if getattr(args, "enable_elastic_ep", False) and args.api_server_count > 1:
            logger.warning(
                "Elastic EP only supports running with with at most one API server. "
                "Capping api_server_count from %d to 1.",
                args.api_server_count,
            )
            args.api_server_count = 1

        if args.api_server_count < 1:
            run_headless(args)
        elif args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            args.api_server_count = None
            uvloop.run(run_server(args))

    def validate(self, args: argparse.Namespace) -> None:
        """参数校验钩子：交由 cli_args.validate_parsed_serve_args 做合法性检查。"""
        validate_parsed_serve_args(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        """注册 `serve` 子命令的参数解析器，挂上全部 serve/引擎相关参数。"""
        serve_parser = subparsers.add_parser(
            self.name,
            help="Launch a local OpenAI-compatible API server to serve LLM "
            "completions via HTTP.",
            description=DESCRIPTION,
            usage="vllm serve [model_tag] [options]",
        )

        serve_parser = make_arg_parser(serve_parser)
        serve_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return serve_parser


def cmd_init() -> list[CLISubcommand]:
    """返回本模块注册的 CLI 子命令列表（供 cli/main.py 收集）。"""
    return [ServeSubcommand()]


def run_headless(args: argparse.Namespace):
    """以 headless 模式运行：只启动 EngineCore/Worker，不启动任何 API server。

    用于纯计算节点（多节点 PP/TP 的非首节点）或外部负载均衡场景；
    本节点作为 worker 加入分布式组，或拉起本地多个数据并行引擎并监控存活。
    捕获 SIGTERM/SIGINT 做优雅退出。
    """
    if args.api_server_count > 1:
        raise ValueError("api_server_count can't be set in headless mode")

    # Create the EngineConfig.
    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(
        usage_context=usage_context, headless=True
    )

    if engine_args.data_parallel_hybrid_lb:
        raise ValueError("data_parallel_hybrid_lb is not applicable in headless mode")

    parallel_config = vllm_config.parallel_config
    local_engine_count = parallel_config.data_parallel_size_local

    if local_engine_count <= 0:
        raise ValueError("data_parallel_size_local must be > 0 in headless mode")

    shutdown_requested = False

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    if parallel_config.node_rank_within_dp > 0:
        from vllm.version import __version__ as VLLM_VERSION

        # Run headless workers (for multi-node PP/TP).
        host = parallel_config.master_addr
        head_node_address = f"{host}:{parallel_config.master_port}"
        logger.info(
            "Launching vLLM (v%s) headless multiproc executor, "
            "with head node address %s for torch.distributed process group.",
            VLLM_VERSION,
            head_node_address,
        )

        executor = MultiprocExecutor(vllm_config, monitor_workers=False)
        executor.start_worker_monitor(inline=True)
        return

    host = parallel_config.data_parallel_master_ip
    port = parallel_config.data_parallel_rpc_port
    handshake_address = get_tcp_uri(host, port)

    logger.info(
        "Launching %d data parallel engine(s) in headless mode, "
        "with head node address %s.",
        local_engine_count,
        handshake_address,
    )

    # Create the engines.
    engine_manager = CoreEngineProcManager(
        local_engine_count=local_engine_count,
        start_index=vllm_config.parallel_config.data_parallel_rank,
        local_start_index=0,
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=handshake_address,
        executor_class=Executor.get_class(vllm_config),
        log_stats=not engine_args.disable_log_stats,
    )

    try:
        engine_manager.monitor_engine_liveness()
    finally:
        timeout = None
        if shutdown_requested:
            timeout = vllm_config.shutdown_timeout
            logger.info("Waiting up to %d seconds for processes to exit", timeout)
        engine_manager.shutdown(timeout=timeout)
        logger.info("Shutting down.")


def run_multi_api_server(args: argparse.Namespace):
    """启动多个 API server 进程（--api-server-count > 1），共享监听 socket。

    先 launch_core_engines 拉起数据并行引擎（含 coordinator），再用
    APIServerProcessManager fork 出多个共享同一监听端口的 API server 进程
    各自连接对应引擎；最后等待全部完成或失败，收到信号时优雅关闭。
    """
    assert not args.headless
    num_api_servers: int = args.api_server_count
    assert num_api_servers > 0

    if num_api_servers > 1:
        setup_multiprocess_prometheus()

    shutdown_requested = False

    # Catch SIGTERM and SIGINT to allow graceful shutdown.
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.debug("Received %d signal.", signum)
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    listen_address, sock = setup_server(args)

    engine_args = vllm.AsyncEngineArgs.from_cli_args(args)
    engine_args._api_process_count = num_api_servers
    engine_args._api_process_rank = -1

    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if num_api_servers > 1 and envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        raise ValueError(
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING cannot be used with api_server_count > 1"
        )

    executor_class = Executor.get_class(vllm_config)
    log_stats = not engine_args.disable_log_stats

    parallel_config = vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    assert parallel_config.local_engines_only or dp_rank == 0

    api_server_manager: APIServerProcessManager | None = None

    from vllm.v1.engine.utils import get_engine_zmq_addresses

    addresses = get_engine_zmq_addresses(vllm_config, num_api_servers)

    with launch_core_engines(
        vllm_config, executor_class, log_stats, addresses, num_api_servers
    ) as (local_engine_manager, coordinator, addresses, tensor_queue):
        # Construct common args for the APIServerProcessManager up-front.
        stats_update_address = None
        if coordinator:
            stats_update_address = coordinator.get_stats_publish_address()

        # Start API servers.
        api_server_manager = APIServerProcessManager(
            listen_address=listen_address,
            sock=sock,
            args=args,
            num_servers=num_api_servers,
            input_addresses=addresses.inputs,
            output_addresses=addresses.outputs,
            stats_update_address=stats_update_address,
            tensor_queue=tensor_queue,
        )

    # Wait for API servers.
    try:
        wait_for_completion_or_failure(
            api_server_manager=api_server_manager,
            engine_manager=local_engine_manager,
            coordinator=coordinator,
        )
    finally:
        timeout = shutdown_by = None
        if shutdown_requested:
            timeout = vllm_config.shutdown_timeout
            shutdown_by = time.monotonic() + timeout
            logger.info("Waiting up to %d seconds for processes to exit", timeout)

        def to_timeout(deadline: float | None) -> float | None:
            return (
                deadline if deadline is None else max(deadline - time.monotonic(), 0.0)
            )

        api_server_manager.shutdown(timeout=timeout)
        if local_engine_manager:
            local_engine_manager.shutdown(timeout=to_timeout(shutdown_by))
        if coordinator:
            coordinator.shutdown(timeout=to_timeout(shutdown_by))
