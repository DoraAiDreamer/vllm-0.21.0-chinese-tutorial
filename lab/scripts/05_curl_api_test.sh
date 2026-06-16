#!/usr/bin/env bash
# ============================================================
# 实验 5: API Server 快速测试 (curl)
# ============================================================
# 前置: docker compose up -d api-server
# 运行: bash scripts/05_curl_api_test.sh
# ============================================================

set -e
BASE="http://localhost:8000"
MODEL="facebook/opt-125m"

echo "============================================================"
echo "1. 健康检查"
echo "============================================================"
curl -s "${BASE}/health" | python3 -m json.tool 2>/dev/null || echo "Server not ready"
echo -e "\n"

echo "============================================================"
echo "2. 模型列表"
echo "============================================================"
curl -s "${BASE}/v1/models" | python3 -m json.tool 2>/dev/null
echo -e "\n"

echo "============================================================"
echo "3. Text Completion (非流式)"
echo "============================================================"
curl -s "${BASE}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"The capital of China is\",
    \"max_tokens\": 30,
    \"temperature\": 0.0
  }" | python3 -m json.tool 2>/dev/null
echo -e "\n"

echo "============================================================"
echo "4. Text Completion (流式)"
echo "============================================================"
curl -s "${BASE}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"Once upon a time\",
    \"max_tokens\": 40,
    \"temperature\": 0.7,
    \"stream\": true
  }"
echo -e "\n"

echo "============================================================"
echo "5. Chat Completion"
echo "============================================================"
curl -s "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"What is 2+2?\"}
    ],
    \"max_tokens\": 30,
    \"temperature\": 0.0
  }" | python3 -m json.tool 2>/dev/null
echo -e "\n"

echo "============================================================"
echo "6. Temperature 对比"
echo "============================================================"
for TEMP in 0.0 0.5 1.0; do
  echo "--- temperature=${TEMP} ---"
  curl -s "${BASE}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${MODEL}\",
      \"prompt\": \"The meaning of life is\",
      \"max_tokens\": 30,
      \"temperature\": ${TEMP}
    }" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['text'])" 2>/dev/null
done
echo ""

echo "============================================================"
echo "7. 停止词测试"
echo "============================================================"
curl -s "${BASE}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"Count from 1 to 10:\",
    \"max_tokens\": 100,
    \"temperature\": 0.0,
    \"stop\": [\"5\"]
  }" | python3 -m json.tool 2>/dev/null
echo -e "\n"

echo "============================================================"
echo "8. Tokenize 端点"
echo "============================================================"
curl -s "${BASE}/tokenize" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${MODEL}\",
    \"prompt\": \"Hello, how are you?\"
  }" | python3 -m json.tool 2>/dev/null
echo -e "\n"

echo "所有测试完成!"
