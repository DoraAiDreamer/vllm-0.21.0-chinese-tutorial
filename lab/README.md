# vLLM 本地实验环境

> 基于 Docker + CPU 模式，无需 GPU 即可学习 vLLM 核心概念

## 目录结构

```
lab/
├── README.md                  # 本文档
├── docker-compose.yml         # Docker Compose 配置
├── configs/                   # 配置文件
│   ├── vllm_logging.json      # 自定义日志配置
│   └── serve_config.yaml      # vLLM Serve YAML 配置
├── scripts/                   # 实验脚本
│   ├── 01_offline_inference.py     # 离线推理基础
│   ├── 02_api_server_test.py       # API Server 测试
│   ├── 03_logging_experiments.py   # 日志系统探索
│   ├── 04_model_params_explorer.py # 引擎参数深度探索
│   └── 05_curl_api_test.sh         # curl 快速测试
├── data/                      # 运行时数据（自动创建）
│   ├── model_cache/           # HuggingFace 模型缓存
│   └── outputs/               # 实验输出
└── logs/                      # 日志输出（自动创建）
```

## 快速开始

### 1. 拉取镜像

```bash
cd lab/
docker compose pull
```

> 镜像大小约 3-4GB，首次拉取需等待

### 2. 启动 API Server

```bash
# 启动（首次会自动下载模型 ~500MB）
docker compose up -d api-server

# 查看启动日志（等待模型加载完成）
docker compose logs -f api-server

# 检查服务状态
curl http://localhost:8000/health
```

### 3. 运行实验脚本

```bash
# 实验 1: 离线推理（最基础）
docker compose run offline-inference /app/scripts/01_offline_inference.py

# 实验 2: API Server 测试（需先启动 api-server）
pip install openai
python scripts/02_api_server_test.py

# 实验 3: 日志系统探索
docker compose run -e VLLM_LOGGING_LEVEL=DEBUG offline-inference /app/scripts/03_logging_experiments.py

# 实验 4: 引擎参数深度探索
docker compose run offline-inference /app/scripts/04_model_params_explorer.py

# 实验 5: curl 快速测试（需先启动 api-server）
bash scripts/05_curl_api_test.sh
```

### 4. 停止服务

```bash
docker compose down
```

## 实验列表

| # | 实验 | 学习目标 | 运行方式 |
|---|------|----------|----------|
| 1 | [离线推理基础](scripts/01_offline_inference.py) | LLM 类用法、SamplingParams、temperature/top_p/top_k 对比 | `docker compose run offline-inference /app/scripts/01_offline_inference.py` |
| 2 | [API Server 测试](scripts/02_api_server_test.py) | OpenAI 兼容接口、流式输出、Chat Completion | `python scripts/02_api_server_test.py`（需先启动 api-server） |
| 3 | [日志系统探索](scripts/03_logging_experiments.py) | 日志环境变量、级别差异、自定义配置 | `docker compose run -e VLLM_LOGGING_LEVEL=DEBUG offline-inference /app/scripts/03_logging_experiments.py` |
| 4 | [引擎参数探索](scripts/04_model_params_explorer.py) | VllmConfig 结构、max_model_len/block_size 影响、KV Cache | `docker compose run offline-inference /app/scripts/04_model_params_explorer.py` |
| 5 | [curl 快速测试](scripts/05_curl_api_test.sh) | REST API 快速验证 | `bash scripts/05_curl_api_test.sh` |

## Docker Compose 服务说明

| 服务 | 用途 | 启动方式 |
|------|------|----------|
| `api-server` | OpenAI 兼容 API 服务器（常驻） | `docker compose up -d api-server` |
| `offline-inference` | 离线推理脚本运行器（按需） | `docker compose run offline-inference python <脚本>` |
| `debug-shell` | 调试 Shell（按需） | `docker compose run debug-shell bash` |

> `offline-inference` 和 `debug-shell` 使用 `profiles: ["run"]` / `profiles: ["debug"]`，
> 不会随 `docker compose up -d` 自动启动，需显式调用。

## 使用的模型

**facebook/opt-125m** — Meta OPT 系列 125M 参数模型

- 参数量: 125M（约 250MB）
- 架构: 标准 Transformer Decoder-only
- 特点: 极小、下载快、CPU 可跑
- 限制: 生成质量有限，仅适合学习 vLLM 用法，不适合评估模型质量

> **如果你有 GPU**，可修改 `docker-compose.yml` 中的模型为更大模型：
> ```yaml
> command: >
>   serve Qwen/Qwen2.5-1.5B-Instruct
>   --tensor-parallel-size 1
>   --gpu-memory-utilization 0.9
>   --dtype auto
>   --max-model-len 2048
> ```
> 并添加 GPU 配置：
> ```yaml
> deploy:
>   resources:
>     reservations:
>       devices:
>         - driver: nvidia
>           count: 1
>           capabilities: [gpu]
> ```

## 常用命令速查

```bash
# ========== 服务管理 ==========
docker compose up -d api-server      # 启动 API Server
docker compose logs -f api-server    # 查看日志
docker compose restart api-server    # 重启
docker compose down                  # 停止所有

# ========== 离线推理 ==========
docker compose run offline-inference /app/scripts/01_offline_inference.py

# ========== 调试 ==========
docker compose run -e VLLM_LOGGING_LEVEL=DEBUG debug-shell bash

# ========== 清理 ==========
docker compose down -v               # 停止并删除 volumes
rm -rf data/model_cache/             # 清理模型缓存（约 500MB）
```

## 国内镜像加速

`docker-compose.yml` 中已配置 `HF_ENDPOINT=https://hf-mirror.com`，使用 HuggingFace 国内镜像加速下载。

如果镜像不可用，可替换为：
```yaml
environment:
  - HF_ENDPOINT=https://hf-mirror.com   # 或 https://huggingface.do.mirr.one
```

## 扩展实验

### 更换模型

编辑 `docker-compose.yml` 中的 `command` 字段，将 `facebook/opt-125m` 替换为：

| 模型 | 大小 | 说明 |
|------|------|------|
| `facebook/opt-125m` | 250MB | 默认，CPU 友好 |
| `facebook/opt-350m` | 700MB | 稍大，CPU 可跑 |
| `facebook/opt-1.3b` | 2.6GB | 需 8GB+ RAM |
| `Qwen/Qwen2.5-0.5B-Instruct` | 1GB | 中文支持好，CPU 可跑 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 3GB | 中文支持好，需较大 RAM |

### 使用 YAML 配置启动

```bash
docker compose run offline-inference bash -c \
  "vllm serve --config /app/configs/serve_config.yaml"
```

### 使用自定义日志配置

```bash
docker compose run \
  -e VLLM_LOGGING_CONFIG_PATH=/app/configs/vllm_logging.json \
  -e VLLM_CONFIGURE_LOGGING=True \
  offline-inference /app/scripts/01_offline_inference.py
```

### 手动进入容器探索

```bash
docker compose run debug-shell bash

# 在容器内
python -c "from vllm import LLM; print('vLLM OK')"
vllm --help
pip list | grep vllm
```

## 注意事项

1. **CPU 模式限制**: 不支持 CUDA Graph、FP8/BF16 量化、FlashAttention
2. **内存需求**: OPT-125M 需约 1GB RAM，OPT-1.3B 需约 4GB RAM
3. **推理速度**: CPU 模式极慢（OPT-125M 约 2-5 tokens/sec），仅适合学习
4. **模型缓存**: 首次下载后模型缓存在 `data/model_cache/`，后续启动秒加载
5. **端口冲突**: 确保 8000 端口未被占用
