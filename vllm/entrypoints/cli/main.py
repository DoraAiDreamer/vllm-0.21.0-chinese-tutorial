# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)


def main():
    """vLLM 命令行工具（`vllm <subcommand> ...`）的统一入口。

    工作流程：
    1. 延迟导入各子命令模块（serve/openai/launch/bench/collect-env/run-batch），
       避免 eager import 导致的依赖/平台问题；
    2. 处理特殊的 ``--omni`` 标志：委托给可选安装的 vllm-omni；
    3. 构建顶层 argparse 解析器，向每个子命令模块收集并注册其子解析器
       （每个子命令是一个 :class:`~vllm.entrypoints.cli.types.CLISubcommand`）；
    4. 解析命令行参数，调用对应子命令的 ``validate`` 做校验，再分发执行
       ``dispatch_function``；未给出子命令时打印帮助。
    """
    import vllm.entrypoints.cli.benchmark.main
    import vllm.entrypoints.cli.collect_env
    import vllm.entrypoints.cli.launch
    import vllm.entrypoints.cli.openai
    import vllm.entrypoints.cli.run_batch
    import vllm.entrypoints.cli.serve
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # 注册全部 CLI 子命令模块；每个模块通过 cmd_init() 返回 CLISubcommand 列表
    CMD_MODULES = [
        vllm.entrypoints.cli.openai,        # vllm serve / api 兼容服务器
        vllm.entrypoints.cli.serve,         # vllm serve（OpenAI API Server）
        vllm.entrypoints.cli.launch,        # 分布式/多节点启动封装
        vllm.entrypoints.cli.benchmark.main,  # vllm bench 性能基准
        vllm.entrypoints.cli.collect_env,   # vllm collect-env 环境信息
        vllm.entrypoints.cli.run_batch,     # vllm run-batch 离线批处理
    ]

    # CLI 环境初始化（日志、环境变量等公共前置）
    cli_env_setup()

    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" in sys.argv:
        # NOTE: Check the spec instead of importing directly here, since things could
        # fail with ImportError due to mismatched versions if things are moved around.
        spec = find_spec("vllm_omni")
        if spec is None:
            logger.error(
                "--omni flag requires a valid instance of vllm-omni to be installed."
            )
            sys.exit(1)

        from vllm_omni.entrypoints.cli.main import main as omni_main

        logger.info("Delegating entrypoint handling to vllm-omni")
        omni_main()
    else:
        # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
        if len(sys.argv) > 1 and sys.argv[1] == "bench":
            logger.debug(
                "Bench command detected, must ensure current platform is not "
                "UnspecifiedPlatform to avoid device type inference error"
            )
            from vllm import platforms

            if platforms.current_platform.is_unspecified():
                from vllm.platforms.cpu import CpuPlatform

                platforms.current_platform = CpuPlatform()
                logger.info(
                    "Unspecified platform detected, switching to CPU Platform instead."
                )

        # 构建顶层 CLI 解析器；FlexibleArgumentParser 支持按 dataclass 自动生成参数
        parser = FlexibleArgumentParser(
            description="vLLM CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        # `vllm -v / --version` 打印版本号
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=importlib.metadata.version("vllm"),
        )
        # 子命令解析器容器（dest="subparser" 记录用户输入的子命令名）
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        # 让每个子命令模块注册自己的子解析器与其处理函数
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                # subparser_init 返回该子命令的 parser；
                # set_defaults 把它的执行入口 cmd.cmd 绑定为 dispatch_function
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        # 若该子命令定义了 validate 钩子，则先做参数校验
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        # 分发到子命令的执行函数；未提供子命令则打印帮助
        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
