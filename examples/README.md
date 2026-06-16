# vLLM Examples

This directory contains examples demonstrating vLLM's various features, from basic inference to advanced deployment scenarios. Examples are organized into the following categories:

## Table of Contents

1. [Basic](#1-basic) — Core offline inference and online serving
2. [Applications](#2-applications) — Real-world application patterns
3. [Generate](#3-generate) — Text and multimodal generation
4. [Pooling](#4-pooling) — Embeddings, classification, scoring, and more
5. [Features](#5-features) — Advanced vLLM capabilities
6. [Disaggregated](#6-disaggregated) — Disaggregated prefill/decode and KV cache sharing
7. [Deployment](#7-deployment) — Production deployment (Kubernetes, SageMaker)
8. [Observability](#8-observability) — Monitoring, metrics, and tracing
9. [Tool Calling](#9-tool-calling) — Function calling and tool use
10. [Reasoning](#10-reasoning) — Reasoning model inference
11. [Speech to Text](#11-speech-to-text) — Speech transcription and translation
12. [RL](#12-rl) — Reinforcement learning (RLHF)
13. [Ray Serving](#13-ray-serving) — Ray-based distributed serving
14. [Templates](#15-jinja-templates) — Chat template files

---

## 1. Basic

### `basic/offline_inference/` — Local model inference

Run models directly on your machine without starting a server.

| File | Description |
|------|-------------|
| `basic.py` | **Entry point** — simplest way to use vLLM. Ideal for beginners. |
| `generate.py` | Text generation with sampling params (`max_tokens`, `temperature`, `top_p`, `top_k`). |
| `chat.py` | Multi-turn chat inference using the `LLM` class. |
| `embed.py` | Generate text embeddings (dense vector representations). |
| `classify.py` | Text classification using model pooling layers. |
| `score.py` | Scoring/ranking tasks (e.g., sentence similarity). |

**Key features demonstrated:** GGUF quantized models, CPU offload (`--cpu-offload-gb`), custom generation configs.

**Quick start:**
```bash
python examples/basic/offline_inference/basic.py
python examples/basic/offline_inference/generate.py --model meta-llama/Llama-2-13b-chat-hf
```

### `basic/online_serving/` — OpenAI-compatible API clients

Clients that connect to a running vLLM server via the OpenAI API.

| File | Description |
|------|-------------|
| `openai_chat_completion_client.py` | Chat completion client (conversational API). |
| `openai_completion_client.py` | Text completion client (continuation API). |

**Prerequisite:** Start a server first: `vllm serve <model_name>`

---

## 2. Applications

### `applications/chatbot/` — Chatbot frontends

| File | Description |
|------|-------------|
| `gradio_webserver.py` | Gradio-based chat web UI (local model). |
| `gradio_openai_chatbot_webserver.py` | Gradio UI connecting to OpenAI-compatible server. |
| `streamlit_openai_chatbot_webserver.py` | Streamlit-based chat UI via OpenAI API. |
| `api_client.py` | Raw API client for programmatic access. |

### `applications/rag/` — Retrieval-Augmented Generation

| File | Description |
|------|-------------|
| `retrieval_augmented_generation_with_langchain.py` | RAG pipeline using LangChain + vLLM as the LLM backend. |
| `retrieval_augmented_generation_with_llamaindex.py` | RAG pipeline using LlamaIndex + vLLM. |

---

## 3. Generate

Text and multimodal generation examples.

| File | Description |
|------|-------------|
| `batched_chat_completions_online.py` | Batched chat completion requests via online API. |
| `qwen_1m_offline.py` | Generating with ultra-long contexts (1M tokens) using Qwen models. |
| `token_generation_client.py` | Token-level generation client. |

### `generate/multimodal/` — Multimodal models

| File | Description |
|------|-------------|
| `vision_language_offline.py` | Vision-language model inference (image + text). |
| `vision_language_multi_image_offline.py` | Multi-image input support. |
| `encoder_decoder_multimodal_offline.py` | Encoder-decoder multimodal models. |
| `audio_language_offline.py` | Audio + text model inference. |
| `mistral-small_offline.py` | Mistral Small vision-language model. |
| `openai_chat_completion_client_for_multimodal.py` | Multimodal via OpenAI API. |
| `qwen2_5_omni/` | Qwen2.5-Omni (full modalities: text, image, audio, video). |
| `qwen3_omni/` | Qwen3-Omni examples (thinker-only mode). |

---

## 4. Pooling

Examples for models with pooling layers: embeddings, classification, scoring, reranking, token-level tasks.

### `pooling/embed/` — Embeddings

| File | Description |
|------|-------------|
| `openai_embedding_client.py` | Embedding generation via OpenAI API. |
| `embed_jina_embeddings_v3_offline.py` | Jina Embeddings v3 offline inference. |
| `embed_matryoshka_fy_offline.py` | Matryoshka embeddings (adaptive dimension). |
| `vision_embedding_offline.py` / `vision_embedding_online.py` | Vision embedding models. |
| `embedding_requests_base64_online.py` | Base64-encoded image inputs. |
| `embedding_requests_bytes_online.py` | Raw byte image inputs. |
| `openai_embedding_long_text/` | Long-text embedding service (client + service setup). |
| `openai_embedding_matryoshka_fy_client.py` | Matryoshka embedding via OpenAI API. |
| `plugin/prithvi_geospatial_mae_*.py` | Geospatial MAE model plugin (satellite imagery embeddings). |

### `pooling/classify/` — Text/Visual Classification

| File | Description |
|------|-------------|
| `classification_online.py` | Text classification via online API. |
| `vision_classification_online.py` | Image classification via online API. |

### `pooling/reward/` — Reward Models

| File | Description |
|------|-------------|
| `sequence_reward_offline.py` / `sequence_reward_online.py` | Sequence-level scoring (RLHF reward). |
| `token_reward_offline.py` / `token_reward_online.py` | Token-level reward scoring. |

### `pooling/score/` — Reranking / Scoring

| File | Description |
|------|-------------|
| `cohere_rerank_client.py` | Cohere reranker via OpenAI API. |
| `qwen3_reranker_offline.py` / `qwen3_reranker_online.py` | Qwen3 reranker models. |
| `colbert_rerank_online.py` | ColBERT-style reranking. |
| `colmodernvbert_rerank_online.py` | ColBERT + ModernBERT reranking. |
| `colqwen3_rerank_online.py` / `colqwen3_5_rerank_online.py` | ColQwen3 reranking variants. |
| `vision_reranker_offline.py` / `vision_rerank_api_online.py` | Vision reranking (image + text). |
| `score_api_online.py` / `rerank_api_online.py` | Generic score/rerank API clients. |
| `using_template_offline.py` / `using_template_online.py` | Template-based scoring. |
| `convert_model_to_seq_cls.py` | Convert cross-encoder to sequence classification. |

### `pooling/token_classify/` — Token-level Classification

| File | Description |
|------|-------------|
| `ner_offline.py` / `ner_online.py` | Named Entity Recognition (NER). |
| `forced_alignment_offline.py` | Forced phoneme/text alignment. |

### `pooling/token_embed/` — Token-level Embeddings

| File | Description |
|------|-------------|
| `jina_embeddings_v4_offline.py` | Jina Embeddings v4 token embeddings. |
| `jina_reranker_v3_offline.py` | Jina Reranker v3 token-level. |
| `multi_vector_retrieval_offline.py` / `multi_vector_retrieval_online.py` | Multi-vector retrieval (DPR-style). |
| `colqwen3_token_embed_online.py` | ColQwen3 token embeddings. |

---

## 5. Features

Advanced vLLM features and configurations.

### `features/lora/` — LoRA Adapters

| File | Description |
|------|-------------|
| `multilora_offline.py` | Multiple LoRA adapters in a single inference. |
| `lora_with_quantization_offline.py` | LoRA with quantized base models (FP8/GPTQ etc.). |

### `features/speculative_decoding/` — Speculative Decoding

| File | Description |
|------|-------------|
| `spec_decode_offline.py` | Draft-model speculative decoding (speedup). |
| `mlpspeculator_offline.py` | MLP speculator (lightweight draft model). |
| `extract_hidden_states_offline.py` | Extract hidden states for speculator training. |

### `features/structured_outputs/` — Structured Outputs

| File | Description |
|------|-------------|
| `structured_outputs_offline.py` | Constrained decoding: JSON schema, regex, grammar. |
| `structured_outputs_client.py` | Online client for structured outputs. |

Supports `structural_tag`, `regex`, JSON schema constraints, and reasoning model output parsing.

### `features/automatic_prefix_caching/` — Prefix Caching

| File | Description |
|------|-------------|
| `prefix_caching_offline.py` | Manual KV cache prefix caching. |
| `automatic_prefix_caching_offline.py` | Automatic prefix caching (built-in). |

### `features/data_parallel/` — Data Parallelism

| File | Description |
|------|-------------|
| `data_parallel_offline.py` | Single-process data parallel inference. |
| `multi_instance_data_parallel.py` | Multi-process/multi-node data parallel. |

### `features/torchrun/` — TorchRun Distributed

| File | Description |
|------|-------------|
| `torchrun_example_offline.py` | Distributed inference via `torchrun`. |
| `torchrun_dp_example_offline.py` | TorchRun data parallel example. |

### `features/prompt_embed/` — Prompt Embeddings

| File | Description |
|------|-------------|
| `prompt_embed_offline.py` | Pre-computed prompt embedding inference (skip tokenization). |
| `prompt_embed_inference_with_openai_client.py` | Prompt embeddings via OpenAI API. |

### `features/pause_resume/` — Pause/Resume

| File | Description |
|------|-------------|
| `pause_resume_offline.py` | Pause and resume inference engine. |
| `data_parallel_pause_resume.py` | Pause/resume in data parallel mode. |

### `features/profiling/` — Profiling

| File | Description |
|------|-------------|
| `simple_profiling_offline.py` | Basic performance profiling. |
| `run_one_batch_offline.py` | Single-batch profiling. |

### `features/batch_invariance/` — Batch Invariance

| File | Description |
|------|-------------|
| `reproducibility_offline.py` | Verify batch-size-invariant results (reproducibility). |

### `features/context_extension/` — Context Extension

| File | Description |
|------|-------------|
| `context_extension_offline.py` | Extend context window beyond training length. |

### `features/reset_kv/` — KV Cache Reset

| File | Description |
|------|-------------|
| `reset_kv_offline.py` | Reset KV cache mid-generation. |

### `features/sharded_state/` — Sharded Model State

| File | Description |
|------|-------------|
| `save_sharded_state_offline.py` | Save model weights in sharded format. |
| `load_sharded_state_offline.py` | Load sharded model state (checkpoint/resume). |

### `features/logits_processor/` — Custom Logits Processing

| File | Description |
|------|-------------|
| `custom.py` | Custom logits processor (modify token probabilities). |
| `custom_req.py` | Request-level custom logits processing. |
| `custom_req_init.py` | Custom logits processing at request init. |

### `features/kv_events/` — KV Cache Events

| File | Description |
|------|-------------|
| `kv_events_subscriber.py` | Subscribe to KV cache lifecycle events. |

### `features/openai_batch/` — OpenAI Batch API

| File | Description |
|------|-------------|
| `openai_example_batch.jsonl` | Example batch request file. |

Run offline batch inference using the OpenAI batch file format (`jsonl`). Supports `/v1/chat/completions`, `/v1/embeddings`, and `/v1/score` endpoints. Can read from local files, HTTP URLs, or AWS S3 (via presigned URLs).

Usage:
```bash
python -m vllm.entrypoints.openai.run_batch -i requests.jsonl -o results.jsonl --model <model>
```

### `features/tensorize_vllm_model.py` — Model Tensorization

Serialize vLLM models to tensor format for deployment.

### `features/logging_configuration.md` — Logging Configuration

How to configure vLLM's logging (levels, formats, handlers).

---

## 6. Disaggregated

Disaggregated prefill/decode architecture: split the LLM inference pipeline into separate prefill-only and decode-only servers for higher throughput. Requires [LMCache](https://github.com/LMCache/LMCache) for KV cache transfer.

### `disaggregated/` — Core Examples

| File | Description |
|------|-------------|
| `disaggregated_prefill.py` | Basic disaggregated prefill example. |

### `disaggregated/disaggregated_encoder/` — Encoder-side Disaggregation

| File | Description |
|------|-------------|
| `disagg_epd_example.sh` / `disagg_1e1pd_example.sh` | Encoder-decoder disaggregation scripts. |
| `disagg_epd_proxy.py` | Proxy server for encoder-decoder disaggregation. |

### `disaggregated/disaggregated_serving/` — Serving-mode Disaggregation

| File | Description |
|------|-------------|
| `disagg_proxy_demo.py` | Disaggregated serving proxy demo. |
| `disagg_proxy_multiturn.py` | Multi-turn conversation with disaggregation. |
| `example_mm_serve.py` | Multimodal serving with disaggregation. |
| `moriio_toy_proxy_server.py` | Toy proxy server for testing. |

### `disaggregated/lmcache/` — LMCache Integration

| File | Description |
|------|-------------|
| `disagg_prefill_lmcache_v1/` | Full LMCache v1 setup: proxy server, configs, launcher scripts. |
| `disagg_prefill_lmcache_v0.py` | LMCache v0 (legacy) disaggregated prefill. |
| `cpu_offload_lmcache.py` | CPU offloading via LMCache (v0 and v1). |
| `kv_cache_sharing_lmcache_v1.py` | KV cache sharing across vLLM instances. |

### `disaggregated/example_connector/` — Custom KV Connectors

| File | Description |
|------|-------------|
| `prefill_example.py` / `decode_example.py` | Custom prefill/decode connector implementation. |
| `run.sh` | Runner script. |

### `disaggregated/flexkv_connector/` — FlexKV

| File | Description |
|------|-------------|
| `prefix_caching_flexkv.py` | Prefix caching with FlexKV connector. |

### `disaggregated/kv_load_failure_recovery_offline/` — KV Load Failure Recovery

| File | Description |
|------|-------------|
| `load_recovery_example_connector.py` | KV cache transfer failure recovery. |
| `prefill_example.py` / `decode_example.py` | Recovery-aware prefill/decode. |

### `disaggregated/mooncake_connector/` — Mooncake Connector

| File | Description |
|------|-------------|
| `mooncake_connector_proxy.py` | Mooncake-based KV cache transfer proxy. |

### `disaggregated/p2p_nccl_xpyd/` — P2P NCCL

| File | Description |
|------|-------------|
| `disagg_proxy_p2p_nccl_xpyd.py` | Peer-to-peer NCCL-based KV transfer. |

---

## 7. Deployment

Production deployment configurations.

| File | Description |
|------|-------------|
| `llm_engine_example.py` | Programmatic LLMEngine usage (no server). |
| `async_llm_streaming.py` | Async streaming generation. |
| `sagemaker-entrypoint.sh` | AWS SageMaker deployment entrypoint. |
| `chart-helm/` | **Helm chart** for Kubernetes deployment (Deployment, HPA, Service, PVC, Secrets, ConfigMap, PodDisruptionBudget, Jobs). |

---

## 8. Observability

Monitoring and observability for vLLM deployments.

### `observability/metrics/` — Metrics

| File | Description |
|------|-------------|
| `offline.py` | Collect and export vLLM metrics (Prometheus format). |

### `observability/opentelemetry/` — Distributed Tracing

| File | Description |
|------|-------------|
| `dummy_client.py` | OpenTelemetry tracing client example. |

### `observability/prometheus_grafana/` — Prometheus + Grafana

| File | Description |
|------|-------------|
| `docker-compose.yaml` | Full Prometheus + Grafana stack (Docker Compose). |
| `prometheus.yaml` | Prometheus scrape configuration. |

### `observability/dashboards/` — Dashboard Configurations

| Platform | Files | Description |
|----------|-------|-------------|
| **Grafana** | `grafana/*.json` | Performance & query statistics dashboards (JSON format). |
| **Perses** | `perses/*.yaml` | Equivalent dashboards in Perses YAML format. |

---

## 9. Tool Calling

Function calling and tool use with LLMs.

| File | Description |
|------|-------------|
| `chat_with_tools_offline.py` | Offline tool calling (local model). |
| `openai_chat_completion_client_with_tools.py` | Tool calling via OpenAI API (optional tools). |
| `openai_chat_completion_client_with_tools_required.py` | Tool calling via OpenAI API (required tools). |
| `openai_chat_completion_client_with_tools_xlam.py` | XLAM tool calling format. |
| `openai_chat_completion_client_with_tools_xlam_streaming.py` | XLAM tool calling with streaming. |
| `openai_responses_client_with_tools.py` | OpenAI Responses API with tools. |
| `openai_responses_client_with_mcp_tools.py` | OpenAI Responses API with MCP (Model Context Protocol) tools. |

---

## 10. Reasoning

Reasoning model (R1-style) inference.

| File | Description |
|------|-------------|
| `openai_chat_completion_with_reasoning.py` | Reasoning model chat completion. |
| `openai_chat_completion_with_reasoning_streaming.py` | Streaming reasoning output. |
| `openai_chat_completion_tool_calls_with_reasoning.py` | Tool calling + reasoning combined. |
| `openai_responses_client.py` | OpenAI Responses API for reasoning models. |

---

## 11. Speech to Text

Speech transcription and translation.

| File | Description |
|------|-------------|
| `openai/openai_transcription_client.py` | Audio transcription via OpenAI API. |
| `openai/openai_translation_client.py` | Audio translation (non-English to English). |
| `lid/openai_lid_client.py` | Language identification + transcription. |
| `realtime/openai_realtime_client.py` | Real-time audio streaming (OpenAI realtime API). |
| `realtime/openai_realtime_microphone_client.py` | Real-time client using microphone input. |

---

## 12. RL

Reinforcement learning (RLHF) training with vLLM.

| File | Description |
|------|-------------|
| `rlhf_async_new_apis.py` | Async RLHF with new APIs. |
| `rlhf_ipc.py` / `rlhf_nccl.py` | RLHF via IPC / NCCL communication. |
| `rlhf_http_ipc.py` / `rlhf_http_nccl.py` | RLHF via HTTP + IPC/NCCL. |
| `rlhf_nccl_fsdp_ep.py` | RLHF with FSDP + Expert Parallelism. |
| `routed_experts_e2e.py` | Routed experts end-to-end example. |
| `skip_loading_weights_in_engine_init.py` | Skip weight loading during engine init (load from existing process). |

---

## 13. Ray Serving

Distributed serving via Ray.

| File | Description |
|------|-------------|
| `ray_serve_deepseek.py` | DeepSeek model serving via Ray. |
| `batch_llm_inference.py` | Batched LLM inference with Ray. |
| `multi-node-serving.sh` / `run_cluster.sh` | Multi-node Ray cluster setup scripts. |
| `elastic_ep/` | Elastic expert parallelism: horizontal scaling of MoE experts. |

---

## 14. Jinja Templates

Chat template files for various model families (used by `tokenizer.chat_template`).

| Template | Model Family |
|----------|-------------|
| `template_alpaca.jinja` | Alpaca |
| `template_baichuan.jinja` | Baichuan |
| `template_chatglm.jinja` / `template_chatglm2.jinja` | ChatGLM / ChatGLM2 |
| `template_chatml.jinja` | ChatML (generic) |
| `template_falcon.jinja` / `template_falcon_180b.jinja` | Falcon |
| `template_inkbot.jinja` | Inkbot assistant |
| `template_teleflm.jinja` | TeleFLM |

### Tool Chat Templates

| Template | Model Family |
|----------|-------------|
| `tool_chat_template_deepseekr1.jinja` | DeepSeek R1 |
| `tool_chat_template_deepseekv3.jinja` / `tool_chat_template_deepseekv31.jinja` | DeepSeek V3 / V3.1 |
| `tool_chat_template_functiongemma.jinja` | Gemma (function calling) |
| `tool_chat_template_gemma3_pythonic.jinja` / `tool_chat_template_gemma4.jinja` | Gemma 3 / 4 (Pythonic) |
| `tool_chat_template_glm4.jinja` | GLM-4 |
| `tool_chat_template_granite.jinja` / `tool_chat_template_granite_20b_fc.jinja` | Granite |
| `tool_chat_template_hermes.jinja` | Hermes (NousResearch) |
| `tool_chat_template_hunyuan_a13b.jinja` | Hunyuan A13B |
| `tool_chat_template_internlm2_tool.jinja` | InternLM2 Tool |
| `tool_chat_template_llama3.1_json.jinja` / `tool_chat_template_llama3.2_json.jinja` | Llama 3.1 / 3.2 (JSON mode) |
| `tool_chat_template_llama3.2_pythonic.jinja` / `tool_chat_template_llama4_json.jinja` / `tool_chat_template_llama4_pythonic.jinja` | Llama 3.2 / 4 |
| `tool_chat_template_minimax_m1.jinja` | MiniMax M1 |
| `tool_chat_template_mistral.jinja` / `tool_chat_template_mistral3.jinja` / `tool_chat_template_mistral_parallel.jinja` | Mistral variants |
| `tool_chat_template_phi4_mini.jinja` | Phi-4 Mini |
| `tool_chat_template_qwen3coder.jinja` | Qwen3 Coder |
| `tool_chat_template_toolace.jinja` | ToolAce |
| `tool_chat_template_xlam_llama.jinja` / `tool_chat_template_xlam_qwen.jinja` | XLAM format |
