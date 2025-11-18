#!/usr/bin/env bash
# Benchmark ShareGPT workload under LRU / LFU / TLru eviction policies.
#
# Usage:
#   bash scripts/compare_eviction_policies.sh
#   MODEL_PATH=/path/to/model bash scripts/compare_eviction_policies.sh
#
# Requires a running Python environment with the sglang repo checked out locally.

# Using ShareGPT, 
# mean: 291, p90: 718.7, p99: 2147.76

set -euo pipefail

REPO_ROOT="/u/jchen61/sglang"
PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export PYTHONPATH

MODEL_PATH="${MODEL_PATH:-qwen/qwen2.5-0.5b-instruct}"
HOST="0.0.0.0"
PORT="${PORT:-30000}"
LOG_LEVEL="${LOG_LEVEL:-warning}"
DATASET="sharegpt"
NUM_PROMPTS="${NUM_PROMPTS:-1500}"
REQUEST_RATES="${REQUEST_RATES:-5, 10, 20}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results}"
mkdir -p "${RESULT_DIR}"

IFS=',' read -ra REQUEST_RATE_LIST <<< "${REQUEST_RATES}"

TLRU_THRESHOLD="${TLRU_THRESHOLD:-1500}"
TLRU_NEXT_PROMPT_ESTIMATE="${TLRU_NEXT_PROMPT_ESTIMATE:-300}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-20000}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-600}"
WAIT_POLL_INTERVAL="${WAIT_POLL_INTERVAL:-2}"

declare -A POLICY_EXTRA
POLICY_EXTRA[lru]=""
POLICY_EXTRA[lfu]=""
POLICY_EXTRA[tlru]="--tlru-threshold ${TLRU_THRESHOLD} --tlru-next-prompt-estimate ${TLRU_NEXT_PROMPT_ESTIMATE}"

wait_for_server() {
  local deadline=$((SECONDS + WAIT_TIMEOUT))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      echo "Server process exited before becoming ready."
      return 1
    fi
    if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${WAIT_POLL_INTERVAL}"
  done
  return 1
}

start_server() {
  local policy="$1"
  local extra="${POLICY_EXTRA[$policy]}"

  local -a args=(
    --model-path "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --log-level "${LOG_LEVEL}"
    --radix-eviction-policy "${policy}"
  )
  if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
  fi

  local -a extra_args=()
  if [[ -n "${extra}" ]]; then
    read -ra extra_args <<< "${extra}"
  fi

  python3 -m sglang.launch_server \
    "${args[@]}" \
    "${extra_args[@]}" \
    > "${RESULT_DIR}/${policy}_server.log" 2>&1 &
  SERVER_PID=$!
  echo "Started ${policy} server (pid ${SERVER_PID}). Waiting for readiness..."
  if ! wait_for_server; then
    echo "Server failed to start within timeout (policy=${policy}). Check ${RESULT_DIR}/${policy}_server.log."
    stop_server
    exit 1
  fi
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
    unset SERVER_PID
  fi
}

run_benchmark() {
  local policy="$1"
  local rate
  for rate in "${REQUEST_RATE_LIST[@]}"; do
    rate="$(echo "${rate}" | xargs)"
    [[ -z "${rate}" ]] && continue
    local outfile="${RESULT_DIR}/${policy}_${DATASET}_${rate}rps.jsonl"
    echo "Benchmarking ${policy} at ${rate} req/s -> ${outfile}"
    python3 -m sglang.bench_serving \
      --backend sglang \
      --host "127.0.0.1" \
      --port "${PORT}" \
      --dataset-name "${DATASET}" \
      --num-prompts "${NUM_PROMPTS}" \
      --request-rate "${rate}" \
      --output-file "${outfile}" || true
  done
}

cleanup() {
  stop_server
}
trap cleanup EXIT

for policy in tlru; do
  echo "==== Running ${policy} benchmark ===="
  start_server "${policy}"
  run_benchmark "${policy}" || true
  stop_server
done

echo "Benchmark results saved to ${RESULT_DIR}."