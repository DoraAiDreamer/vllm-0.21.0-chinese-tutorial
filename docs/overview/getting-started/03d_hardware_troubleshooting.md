# NVIDIA GPU / 昇腾 910B 硬件部署常见问题与排错

> 基于 vllm-0.21.0 版本

本文档整理 vLLM 在 NVIDIA GPU 和华为昇腾 910B 上部署时遇到的通用硬件相关问题、排错指南及最佳实践。

---

## 一、vLLM 平台支持现状

### 1.1 官方支持的硬件平台

截至 vLLM 0.21.0，`vllm/platforms/` 目录下支持的官方平台：

| 平台 | 枚举值 | 平台文件 | 状态 |
|------|--------|----------|------|
| NVIDIA CUDA | `CUDA` | `cuda.py` | 完整支持 |
| AMD ROCm | `ROCM` | `rocm.py` | 完整支持 |
| Intel XPU | `XPU` | `xpu.py` | 支持 |
| Google TPU | `TPU` | `tpu.py` | 支持 |
| CPU | `CPU` | `cpu.py` | 支持 |
| 其他 (OOT) | `OOT` | — | 社区适配 |

### 1.2 昇腾 910B 的支持状态

**重要：** vLLM 0.21.0 官方代码中 **没有** 独立的 Ascend/NPU 平台实现（无 `vllm/platforms/ascend.py` 或 `npu.py`）。

昇腾 910B 的 vLLM 适配通过以下途径：
1. **华为 vLLM 分支**：华为维护的 fork，包含 NPU 平台适配
2. **社区适配**：基于 vLLM 的 NPU 后端实现
3. **OOT (Out-of-Tree) 插件**：通过插件系统注册自定义平台

因此，在官方 vLLM 0.21.0 上直接运行于 910B 会遇到大量兼容性问题。

---

## 二、NVIDIA GPU 通用问题

### 2.1 显存管理

#### 2.1.1 OOM（显存溢出）

**现象：**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has a total capacity of 79.21 GiB of which 1.89 GiB is free.
```

**排查步骤：**

```python
# 步骤 1：检查 GPU 显存使用
import torch
print(torch.cuda.memory_allocated())  # 已分配
print(torch.cuda.memory_reserved())   # 已保留（含缓存）
print(torch.cuda.max_memory_allocated())  # 峰值
print(torch.cuda.mem_info(0))  # 总显存信息
```

**解决方案（按优先级）：**

| 方案 | 方法 | 效果 |
|------|------|------|
| 1 | 降低 `gpu_memory_utilization` | 减少 KV Cache 空间 |
| 2 | 减小 `max_model_len` | 减少 KV Cache 长度 |
| 3 | 使用量化（FP8/AWQ/GPTQ） | 减少权重显存 |
| 4 | 增加 `tensor_parallel_size` | 权重分片 |
| 5 | 启用 `cpu_offload_gb` | 权重卸载到 CPU |
| 6 | 使用 `enable_prefix_caching=True` | 复用 KV Cache |

```python
# 完整 OOM 修复示例
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    gpu_memory_utilization=0.85,     # 从 0.92 降到 0.85
    max_model_len=4096,               # 限制长度
    quantization="fp8",               # FP8 量化
    tensor_parallel_size=2,           # 2 卡并行
)
```

#### 2.1.2 显存泄漏

**现象：** 长时间推理后显存持续增长，即使没有请求也在增长。

**排查：**
```python
# 检查显存碎片
torch.cuda.memory_summary(device, abbreviated=False)

# 检查缓存
torch.cuda.memory_stats(device)
```

**解决：**
```python
# 方法 1：定期重启引擎
# 在 API Server 中设置请求超时
vllm serve MODEL --request-timeout 600

# 方法 2：减少 CUDA 缓存增长
# 设置 VLLM_WORKER_MULTIPROC_METHOD=spawn（避免 fork 导致的显存泄漏）
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 方法 3：使用 V1 引擎（默认）
# V1 引擎的显存管理比 V0 更严格
```

#### 2.1.3 KV Cache 分配不足

**现象：**
```
WARNING: The model's max seq len (131072) is larger than the maximum number 
of tokens that can be stored in KV cache. Reducing max_model_len to 16384.
```

**原因：** 模型声称的最大长度对应的 KV Cache 超过了可用显存。

**解决：**
```python
# 方法 1：手动设置合理的 max_model_len
llm = LLM(model="model", max_model_len=8192)

# 方法 2：增加 gpu_memory_utilization
llm = LLM(model="model", gpu_memory_utilization=0.95)

# 方法 3：使用 KV Cache 量化
llm = LLM(model="model", kv_cache_dtype="fp8")

# 方法 4：允许设置超长长度（不推荐，可能 OOM）
import os
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
```

### 2.2 GPU 通信问题

#### 2.2.1 NCCL 初始化失败

**现象：**
```
RuntimeError: NCCL error in: ... unhandled error
NCCL version 2.18.x
```

**解决：**
```bash
# 方法 1：设置 NCCL 环境变量
export NCCL_DEBUG=INFO          # 查看详细日志
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
export NCCL_IB_DISABLE=1        # 禁用 InfiniBand（如无）
export NCCL_P2P_DISABLE=1       # 禁用 P2P（如需要）

# 方法 2：使用自定义 all-reduce
llm = LLM(model="model", disable_custom_all_reduce=True)
```

#### 2.2.2 GPU P2P 访问受限

**现象：**
```
RuntimeError: Peer-to-peer access failure is not supported
```

**解决：**
```bash
# 禁用 P2P
export NCCL_P2P_DISABLE=1

# 或使用 NVLINK 替代
export NCCL_P2P_LEVEL=NVL
```

#### 2.2.3 多卡不一致

**现象：** 不同 GPU 上的推理结果不一致。

**排查：**
```bash
# 检查 GPU 间连接
nvidia-smi topo -m
```

输出示例：
```
	GPU0	GPU1	GPU2	GPU3	CPU Affinity
GPU0	 X 	SYS	SYS	SYS	0-15
GPU1	SYS	 X 	SYS	SYS	0-15
GPU2	SYS	SYS	 X 	SYS	16-31
GPU3	SYS	SYS	SYS	 X 	16-31
```

**解决：**
- 确保 GPU 之间通过 NVLINK 或 PCIe 直连
- 使用 `NCCL_P2P_LEVEL=NVL` 优先使用 NVLINK
- 如 GPU 间无直连，使用 `NCCL_P2P_DISABLE=1`

### 2.3 CUDA 版本兼容性

#### 2.3.1 CUDA Toolkit 版本

**vLLM 0.21.0 推荐的 CUDA 版本：**

| PyTorch 版本 | 推荐 CUDA | 最低 CUDA |
|-------------|-----------|-----------|
| 2.5.x | 12.4 | 12.1 |
| 2.4.x | 12.1 | 11.8 |

**检查：**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
```

#### 2.3.2 CUDA Graph 问题

**现象：** CUDA Graph 编译失败或运行时错误。

**解决：**
```python
# 方法 1：禁用 CUDA Graph
llm = LLM(model="model", enforce_eager=True)

# 方法 2：限制 CUDA Graph 捕获的 batch size
llm = LLM(
    model="model",
    compilation_config={
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
    },
)

# 方法 3：使用混合模式（默认）
# CUDA Graph 用于小 batch，eager mode 用于大 batch
```

### 2.4 GPU 类型兼容性

#### 2.4.1 不同 GPU 类型混用

**现象：** 多卡推理时报错或不一致。

**解决：**
```bash
# 确保所有 GPU 型号一致
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# 如混用，指定可见设备
CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py
```

#### 2.4.2 GPU 计算能力

**vLLM 对 GPU 计算能力的要求：**

| GPU | 计算能力 | 支持情况 |
|-----|----------|----------|
| H100/H200 | 9.0 | 完全支持（FP8 原生） |
| A100/A800 | 8.0 | 完全支持 |
| A6000/A40 | 8.6 | 完全支持 |
| RTX 4090/4080 | 8.9 | 完全支持 |
| RTX 3090/3080 | 8.6 | 完全支持 |
| V100 | 7.0 | 部分支持（无 BF16/FP8） |
| T4 | 7.5 | 部分支持（无 BF16） |

```python
# 检查 GPU 计算能力
import torch
cap = torch.cuda.get_device_capability()
print(f"Compute Capability: {cap[0]}.{cap[1]}")
```

---

## 三、昇腾 910B 通用问题

### 3.1 环境准备

#### 3.1.1 驱动和固件

```bash
# 检查驱动版本
npu-smi info

# 检查 NPU 状态
npu-smi info -t usages -i 0

# 检查固件版本
npu-smi info -t fw-version
```

**推荐驱动版本：** Ascend Driver 24.1.RC2+ 或更高

#### 3.1.2 CANN 软件栈

```bash
# 检查 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# 推荐 CANN 版本：8.0.RC2+ 或更高
```

#### 3.1.3 环境变量

```bash
# 基本 NPU 环境变量
export ASCEND_HOME=/usr/local/Ascend
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
export PATH=$ASCEND_HOME/ascend-toolkit/latest/bin:$PATH

# NPU 设备可见性
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# Ray 框架兼容
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# Worker 启动方式（NPU 上推荐 spawn）
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 内存管理
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
```

### 3.2 vLLM 在 910B 上的主要限制

| 限制 | 原因 | 影响 |
|------|------|------|
| 无原生 NPU 平台 | 官方无 ascend.py | 需使用华为适配分支 |
| Triton 不可用 | Triton 仅支持 CUDA | 大部分 kernel 不可用 |
| FlashAttention 不可用 | FA 仅支持 CUDA/ROCm | 需使用 NPU 专用后端 |
| CUDA Graph 不可用 | NPU 不支持 CUDA Graph | 性能下降 |
| FP8 支持有限 | 910B FP8 精度不如 A100 | 量化效果打折 |
| NCCL 不可用 | NPU 使用 HCCL | 分布式通信需适配 |

### 3.3 910B 上 vLLM 的替代方案

#### 方案 1：华为 vLLM 分支

```bash
# 使用华为维护的 vLLM 分支
git clone https://gitee.com/ascend/vllm.git
cd vllm
pip install -e .
```

#### 方案 2：MindIE 推理引擎

```bash
# 华为 MindIE 推理引擎
pip install mindie
```

#### 方案 3：昇腾原生推理框架

```bash
# MindSpore Inference
pip install mindspore
```

### 3.4 910B 上常见错误排查

#### 错误 1：CUDA 初始化失败

```
RuntimeError: Found no NVIDIA driver on your system.
```

**解决：**
```bash
# 确认安装了 Ascend 驱动而非 NVIDIA 驱动
npu-smi info

# 如安装了 NVIDIA 驱动，卸载
sudo apt remove nvidia-*
```

#### 错误 2：NPU 算子编译失败

```
RuntimeError: AicCoreTranslator failed, error name: CompileOperatorFail
```

**解决：**
```bash
# 清理 NPU 缓存
rm -rf ~/.ascend/ascend_pro/

# 升级 CANN 版本
# 参考华为文档升级
```

#### 错误 3：HCCL 通信失败

```
RuntimeError: HCCL error: HCCL_ERROR_COMM_CREATE_FAIL
```

**解决：**
```bash
# 检查 HCCL 配置
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_QUEUE_SIZE=256

# 检查所有 NPU 是否正常
npu-smi info -t list
```

#### 错误 4：显存不足

```
RuntimeError: [Ascend] Failed to allocate memory, size: xxx
```

**解决：**
```bash
# 降低显存使用
export ASCEND_MEMORY_SLOTH=1  # 启用内存延迟分配

# 检查 NPU 显存
npu-smi info -t memory -i 0
```

### 3.5 910B 上模型部署建议

```bash
# 910B 推荐配置
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
```

```python
# 910B 上仅推荐部署小模型
# 需使用华为适配的 vLLM 分支
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="float16",              # 910B 优先 FP16
    enforce_eager=True,           # 禁用 CUDA Graph
    tensor_parallel_size=4,       # 4 卡 910B (64GB)
    gpu_memory_utilization=0.85,  # 保守设置
    max_model_len=4096,           # 保守设置
)
```

---

## 四、NVIDIA GPU 推荐配置速查

### 4.1 按 GPU 型号推荐

| GPU | 显存 | 支持模型 | 关键参数 |
|-----|------|----------|----------|
| RTX 3090/4090 | 24GB | 7B-14B | `dtype="bfloat16"` |
| A10 | 24GB | 7B-14B | `dtype="bfloat16"` |
| A100 40GB | 40GB | 7B-32B | `tp=1-2` |
| A100 80GB | 80GB | 7B-72B | `tp=1-4` |
| H100 80GB | 80GB | 7B-72B | `tp=1-4, fp8` |
| H200 141GB | 141GB | 7B-235B | `tp=1-8, fp8` |
| L40S 48GB | 48GB | 7B-32B | `tp=1-2` |

### 4.2 按模型规模推荐

| 模型规模 | 最小 GPU | 推荐 GPU | TP Size | 关键参数 |
|----------|----------|----------|---------|----------|
| 1-7B | 1x RTX 3090 | 1x RTX 4090 | 1 | `dtype="bfloat16"` |
| 14-32B | 1x A100 80GB | 2x A100 80GB | 1-2 | `dtype="bfloat16"` |
| 70-72B | 2x A100 80GB | 4x A100 80GB | 2-4 | `tp=4` 或 `tp=2,fp8` |
| 100-200B | 4x A100 80GB | 8x H100 | 4-8 | `tp=8, ep=True` |
| 200B+ | 8x H100 | 8x H200 | 8 | `tp=8, ep=True, fp8` |

---

## 五、性能优化

### 5.1 NVIDIA GPU 性能调优

#### 5.1.1 Attention 后端选择

```python
# 自动选择（默认）
llm = LLM(model="model")

# H100/A100 推荐 FlashInfer
llm = LLM(
    model="model",
    attention_config={"backend": "flashinfer"},
)

# 兼容性优先
llm = LLM(
    model="model",
    attention_config={"backend": "flex_attention"},  # PyTorch 原生
)
```

#### 5.1.2 MoE Backend 选择

```python
# H100 + FP8 MoE
llm = LLM(model="model", moe_backend="deep_gemm")

# 通用兼容
llm = LLM(model="model", moe_backend="triton")

# CUTLASS 优化
llm = LLM(model="model", moe_backend="cutlass")
```

#### 5.1.3 torch.compile 优化

```python
# 编译所有支持的 batch size
llm = LLM(model="model", compilation_config=1)

# 指定编译的 batch size
llm = LLM(
    model="model",
    compilation_config={"inductor_compile_size": [1, 2, 4, 8, 16, 32, 64]},
)
```

#### 5.1.4 CUDA Graph 优化

```python
# 自动选择（默认）
llm = LLM(model="model")

# 手动指定捕获的 batch size
llm = LLM(
    model="model",
    compilation_config={
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
    },
)

# 限制最大捕获 batch size
llm = LLM(
    model="model",
    compilation_config={
        "max_cudagraph_capture_size": 128,
    },
)
```

### 5.2 内存优化

#### 5.2.1 KV Cache 量化

```python
# FP8 KV Cache（需要 GPU 支持 FP8）
llm = LLM(model="model", kv_cache_dtype="fp8")

# BF16 KV Cache（默认，比 FP16 精度更好）
llm = LLM(model="model", kv_cache_dtype="bfloat16")

# 自动选择
llm = LLM(model="model", kv_cache_dtype="auto")
```

#### 5.2.2 CPU Offload

```python
# 将模型权重卸载到 CPU
llm = LLM(
    model="model",
    cpu_offload_gb=10,  # 卸载 10GB 到 CPU
)

# Prefetch Offload（更细粒度）
llm = LLM(
    model="model",
    offload_group_size=2,       # 每 2 层一组
    offload_num_in_group=1,     # 每组卸载 1 层
    offload_prefetch_step=2,    # 预取 2 步
    offload_params={"gate_up_proj", "down_proj"},  # 选择性卸载
)
```

#### 5.2.3 Prefix Caching

```python
# 启用前缀缓存（对重复 prompt 场景有效）
llm = LLM(
    model="model",
    enable_prefix_caching=True,
)

# 指定哈希算法
llm = LLM(
    model="model",
    prefix_caching_hash_algo="md5",  # "md5" | "crc32"
)
```

---

## 六、监控与调试

### 6.1 GPU 监控

```bash
# 实时监控 GPU 使用
nvidia-smi dmon -s u

# 监控显存分配
nvidia-smi -l 1

# 查看详细 GPU 信息
nvidia-smi -q
```

### 6.2 vLLM 日志

```python
# 启用详细日志
import vllm.logger
vllm.logger.init_logger("vllm", level="DEBUG")

# 禁用统计日志（生产环境推荐）
llm = LLM(model="model", disable_log_stats=True)

# 禁用请求日志
llm = LLM(model="model", disable_log_requests=True)
```

### 6.3 性能分析

```python
# 启用 profiling
llm = LLM(
    model="model",
    profiler_config={
        "profile_start_step": 10,  # 从第 10 步开始记录
        "profile_steps": 5,        # 记录 5 步
    },
)
```

---

## 七、排错决策树

```
硬件部署问题
    │
    ├── NVIDIA GPU？
    │       ├── OOM？
    │       │       ├── 降低 gpu_memory_utilization
    │       │       ├── 减小 max_model_len
    │       │       ├── 使用量化
    │       │       └── 增加 tensor_parallel_size
    │       │
    │       ├── NCCL 报错？
    │       │       ├── 检查 GPU 间连接 (nvidia-smi topo -m)
    │       │       ├── 设置 NCCL_SOCKET_IFNAME
    │       │       └── 禁用 P2P: NCCL_P2P_DISABLE=1
    │       │
    │       ├── CUDA Graph 报错？
    │       │       ├── enforce_eager=True
    │       │       └── 调整 cudagraph_capture_sizes
    │       │
    │       └── 性能低？
    │               ├── 检查 attention backend
    │               ├── 启用 torch.compile
    │               ├── 启用 CUDA Graph
    │               └── 检查 GPU 间连接质量
    │
    ├── 昇腾 910B？
    │       ├── 使用华为 vLLM 分支
    │       ├── 设置 ASCEND_RT_VISIBLE_DEVICES
    │       ├── 设置 VLLM_WORKER_MULTIPROC_METHOD=spawn
    │       ├── 禁用 CUDA Graph (enforce_eager=True)
    │       ├── 使用 FP16 而非 BF16
    │       └── 保守设置显存 (gpu_memory_utilization=0.85)
    │
    └── AMD GPU？
            ├── 使用 ROCm 平台
            ├── 设置 VLLM_ROCM_USE_AITER_MLA=1
            └── 使用 aiter MoE backend
```
