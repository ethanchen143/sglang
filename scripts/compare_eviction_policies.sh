#!/usr/bin/env bash
# Benchmark ShareGPT workload under LRU / LFU / TLru eviction policies.
#
# Usage:
#   bash scripts/compare_eviction_policies.sh
#   MODEL_PATH=/path/to/model bash scripts/compare_eviction_policies.sh
#
# Requires a running Python environment with the sglang repo checked out locally.

# Using ShareGPT dataset, with 1000 prompts.
# Length - mean: 291, p90: 718.7, p99: 2147.76

# Use Loogle, low qps, so the scheduler wouldnt interfere (High qps -> batching, Low qps -> fewer batching, easier to reason about)
# lower the caching, mem ratio as server arg (select working set), fully associated cache, three types of cache
# 1GB - small, look at hidden size of the model
# --memoery util - (kv cache + model weights) / gpu capacity (A100, 40GB), max memory usage, limit kv cache capactiy
# max_total_tokens is for scheduler

set -euo pipefail

REPO_ROOT="/u/jchen61/sglang"
PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export PYTHONPATH

MODEL_PATH="${MODEL_PATH:-qwen/qwen2.5-0.5b-instruct}"
HOST="0.0.0.0"
PORT="${PORT:-30000}"
LOG_LEVEL="${LOG_LEVEL:-debug}"
DATASET="sharegpt"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
REQUEST_RATES="${REQUEST_RATES:-2,4}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results}"
mkdir -p "${RESULT_DIR}"

# Optional: path to a prepared Loogle dataset (e.g., longdep_qa.json).
# Leave empty to use the default behavior of the HiCache benchmark script.
DATASET_PATH="${DATASET_PATH:-}"

IFS=',' read -ra REQUEST_RATE_LIST <<< "${REQUEST_RATES}"

TLRU_THRESHOLD="${TLRU_THRESHOLD:-500}"
TLRU_NEXT_PROMPT_ESTIMATE="${TLRU_NEXT_PROMPT_ESTIMATE:-300}"

# Control KV cache capacity via static memory fraction instead of explicit token cap.
# mem_fraction_static ~= (model weights + KV cache pool) / GPU memory capacity.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.05}"

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

  # Note: we intentionally do NOT enable hierarchical cache here.
  # This benchmark uses pure GPU radix cache with different eviction policies.
  if [[ -n "${MEM_FRACTION_STATIC}" ]]; then
    args+=(--mem-fraction-static "${MEM_FRACTION_STATIC}")
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

# use hicache benchmark script, but dont enable hicache
# dump out the cache usage (for debug), scheduler_metrics_mixin,

run_benchmark() {
  local policy="$1"
  local rate
  for rate in "${REQUEST_RATE_LIST[@]}"; do
    rate="$(echo "${rate}" | xargs)"
    [[ -z "${rate}" ]] && continue
    local outfile="${RESULT_DIR}/${policy}_${DATASET}_${rate}rps.jsonl"
    echo "Benchmarking ${policy} at ${rate} req/s -> ${outfile}"
    python3 "${REPO_ROOT}/benchmark/hicache/bench_serving.py" \
      --backend sglang \
      --host "127.0.0.1" \
      --port "${PORT}" \
      --model "${MODEL_PATH}" \
      --dataset-name "${DATASET}" \
      --num-prompts "${NUM_PROMPTS}" \
      --request-rate "${rate}" \
      --enable-multiturn \
      --disable-shuffle \
      ${DATASET_PATH:+--dataset-path "${DATASET_PATH}"} \
      --output-file "${outfile}" || true
  done
}

cleanup() {
  stop_server
}
trap cleanup EXIT

for policy in tlru lru; do
  echo "==== Running ${policy} benchmark ===="
  start_server "${policy}"
  run_benchmark "${policy}" || true
  stop_server
done

echo "Benchmark results saved to ${RESULT_DIR}."