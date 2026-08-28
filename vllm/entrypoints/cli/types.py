# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import typing

# 此处 FlexibleArgumentParser 根据实际导入路径调整
if typing.TYPE_CHECKING:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
else:
    FlexibleArgumentParser = argparse.ArgumentParser


class CLISubcommand:
    """Base class for CLI argument handlers.

    命令行子命令处理器的基类。

    vLLM 所有顶层子命令（serve、llm、benchmark 等）都应当继承该基类。
    子类必须重写 :meth:`cmd` 和 :meth:`subparser_init`。
    参数校验逻辑可选择在 :meth:`validate` 钩子函数中实现。
    """

    # 在命令行暴露出来的子命令名称，例如 ``"serve"``
    name: str

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        """执行当前子命令的业务逻辑。

        Args:
            args: 解析完成后的命令行参数命名空间对象。

        Raises:
            NotImplementedError: 子类没有重写该方法时抛出异常。
        """
        raise NotImplementedError("Subclasses should implement this method")

    def validate(self, args: argparse.Namespace) -> None:
        """在正式执行命令前，校验解析后的命令行参数。

        可重写此钩子，用来检测冲突参数、非法配置。
        默认实现不执行任何校验。

        Args:
            args: 解析完成后的命令行参数命名空间对象。
        """
        pass

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        """为当前子命令注册命令行参数定义。

        添加启动选项、帮助说明、尾部提示文本等配置。

        Args:
            subparsers: argparse 顶层子命令容器对象。

        Returns:
            配置完成后的子解析器实例。

        Raises:
            NotImplementedError: 子类没有重写该方法时抛出异常。
        """
        raise NotImplementedError("Subclasses should implement this method")
