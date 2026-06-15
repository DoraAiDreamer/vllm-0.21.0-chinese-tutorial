# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For contribution policy, commit conventions, and AI-assisted PR rules, see [AGENTS.md](AGENTS.md).

## Architecture Overview

vLLM is a high-throughput LLM inference and serving library. The codebase is organized into:

- **`vllm/`** — Pure Python package. The runtime core.
  - **`vllm/v1/`** — The new v1 execution engine (now the default). Contains its own LLMEngine, executor, scheduler, pool, sampler, and speculative decoding modules. The top-level `vllm.engine` and `vllm.entrypoints.llm` simply alias to the v1 implementations.
  - **`vllm/engine/`** — Legacy v0 engine (largely superseded by v1).
  - **`vllm/model_executor/models/`** — 290+ model architecture implementations supporting decoder-only LLMs, MoE, hybrid attention/state-space, multimodal, embedding, reward, and classification models.
  - **`vllm/kernels/`** — Triton-based kernels (e.g., `qkv_padded_fp8_quant.py`) and the `vllm_c` custom C++ extension wrapper.
  - **`vllm/compilation/`** — `torch.compile` integration: CUDA graph capture, piecewise backend, static graph compilation, codegen, and compiler passes.
  - **`vllm/lora/`** — Multi-LoRA support for dense and MoE layers (model wrapper, request handling, worker manager, ops).
  - **`vllm/entrypoints/`** — Public APIs: OpenAI-compatible server, CLI, LLM class, gRPC server, Anthropic API, MCP, and chat utilities.
  - **`vllm/config/`** — Configuration dataclasses for every subsystem (attention, compilation, device, distributed parallelism, quantization, model, scheduler, etc.).
  - **`vllm/inputs/`** — Input parsing: TextPrompt, TokensPrompt, PromptType, and multimodal input processing.
  - **`vllm/multimodal/`** — Multimodal model support (vision, audio, etc.).
  - **`vllm/distributed/`** — Tensor/pipeline/data/expert/context parallelism, weight transfer, KV connectors.
  - **`vllm/quantization/`** — Quantization backend plugins (FP8, INT8, GPTQ, AWQ, GGUF, etc.).
  - **`vllm/third_party/`** — Third-party hardware integrations (excluded from linting).
  - **`vllm/transformers_utils/`** — Hugging Face integration (tokenizer, model loading).

- **`csrc/`** — CUDA/C++ native extensions built via CMake + PyTorch cpp_extension.
  - `attention/` — Attention kernels (FlashAttention, FlashInfer integrations).
  - `moe/` — MoE routing and topk kernels.
  - `mamba/` — Mamba/state-space model kernels.
  - `cache_kernels/` — KV cache management operations.
  - `layernorm_kernels/` — Layer normalization kernels.
  - `quantization/` — Quantization kernels (FP8, etc.).
  - `torch_bindings.cpp` — Python bindings for the native library.
  - `cpu/` — CPU-specific kernels (sgl-kernels).

## Development Workflow

### Environment setup (Mandatory)

```bash
# Always use `uv`, never system python3 or bare pip
uv venv --python 3.12
source .venv/bin/activate

# Install pre-commit
uv pip install -r requirements/lint.txt
pre-commit install
```

### Building and installing

```bash
# Python-only changes (uses precompiled wheels for torch)
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Full build (also compiles csrc/ C++/CUDA code)
uv pip install -e . --torch-backend=auto
```

### Linting

```bash
# Run all pre-commit hooks on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff-check --all-files

# mypy (requires manual stage)
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

Pre-commit hooks: ruff-check + ruff-format, typos, clang-format (for csrc/), markdownlint, actionlint, pip-compile.

### Testing

```bash
# Install test dependencies (x86_64)
uv pip install -r requirements/test/cuda.txt
# Or on other platforms:
uv pip install -r requirements/test/cuda.in

# Run a specific test
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# v1-specific tests live under tests/v1/
```

### Key pytest markers

- `slow_test` — Slow tests, skip in fast CI
- `core_model` — Model tests that should run in each PR
- `distributed` — Distributed GPU tests only
- `cpu_test` — CPU-only tests
- `hybrid_model` — Models with Mamba layers
- `optional` — Need `--optional` flag to run

## Important Conventions

- **Never use system `python3` or bare `pip`/`pip install`.** Always go through `uv` and `.venv/bin/python`.
- **v1 is the default.** The top-level `vllm.engine` and `vllm.entrypoints.llm` are thin aliases to `vllm.v1.*`. New work should target v1.
- **`vllm/third_party/`** is excluded from all linting.
- **Build system:** CMake + PyTorch cpp_extension for `csrc/`. Build config is in `CMakeLists.txt` and `setup.py`. Target device is auto-detected from torch (cuda/rocm/xpu/cpu).
- **Package metadata:** `pyproject.toml` uses setuptools-scm for versioning. ruff config is inline. mypy uses `pydantic.mypy` plugin.
- **Models:** Each model file in `vllm/model_executor/models/` defines a `ModelHandler` class following the vLLM model registry pattern. Register via the ModelRegistry.
