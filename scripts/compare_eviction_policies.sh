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

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
HOST="0.0.0.0"
PORT="${PORT:-30000}"
LOG_LEVEL="${LOG_LEVEL:-debug}"
DATASET="sharegpt"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
REQUEST_RATES="${REQUEST_RATES:-4,8,16}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmark_results_1500}"
mkdir -p "${RESULT_DIR}"
BASE_RESULT_DIR="${RESULT_DIR}"
# Control KV cache capacity via a fixed set of memory fractions instead of an external argument.
# mem_fraction ~= (model weights + KV cache pool) / GPU memory capacity.
# Update this list to change the sweep.
MEM_FRACTION_LIST=(0.6 0.75 0.9)

# Optional: path to a prepared Loogle dataset (e.g., longdep_qa.json).
# Leave empty to use the default behavior of the HiCache benchmark script.
DATASET_PATH="${DATASET_PATH:-}"

IFS=',' read -ra REQUEST_RATE_LIST <<< "${REQUEST_RATES}"

TLRU_THRESHOLD="${TLRU_THRESHOLD:-1500}"
TLRU_NEXT_PROMPT_ESTIMATE="${TLRU_NEXT_PROMPT_ESTIMATE:-300}"

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
  local mem_fraction="$2"
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
  if [[ -n "${mem_fraction}" ]]; then
    args+=(--mem-fraction-static "${mem_fraction}")
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
      --fixed-output-len 512 \
      --max-concurrency 16 \
      --request-rate "${rate}" \
      --enable-multiturn \
      --disable-shuffle \
      ${DATASET_PATH:+--dataset-path "${DATASET_PATH}"} \
      --output-file "${outfile}" || true
  done
}

summarize_hit_rate() {
  # Aggregate KV cache hit rate from scheduler logs.
  # We parse the "Prefill batch" log lines emitted by `log_prefill_stats`
  # in `scheduler_metrics_mixin.py`, which contain "#new-token" and
  # "#cached-token" fields.
  local policy="$1"
  local logfile="${RESULT_DIR}/${policy}_server.log"
  local outfile="${RESULT_DIR}/${policy}_hit_rate.txt"

  if [[ ! -f "${logfile}" ]]; then
    echo "No server log found for policy ${policy} at ${logfile}" >&2
    return
  fi

  echo "Summarizing cache hit rate for policy ${policy} -> ${outfile}"
  grep "Prefill batch" "${logfile}" | awk '
    {
      if (match($0, /#new-token: ([0-9]+)/, a) && match($0, /#cached-token: ([0-9]+)/, b)) {
        new = a[1]
        cached = b[1]
        total_new += new
        total_cached += cached
      }
    }
    END {
      total = total_new + total_cached
      if (total > 0) {
        rate = total_cached / total
        printf "prefill_cache_hit_rate=%.6f\tcached_tokens=%d\ttotal_tokens=%d\n", rate, total_cached, total
      } else {
        print "prefill_cache_hit_rate=0"
      }
    }
  ' > "${outfile}" || true
}

add_hit_rate_to_jsonl_files() {
  # Parse the summarized hit rate and inject it into all JSONL result files
  # for the given policy in the current RESULT_DIR.
  local policy="$1"
  local summary_file="${RESULT_DIR}/${policy}_hit_rate.txt"

  if [[ ! -f "${summary_file}" ]]; then
    echo "No hit-rate summary for policy ${policy} at ${summary_file}" >&2
    return
  fi

  local hit_rate
  hit_rate="$(awk -F'[=\t]' '/prefill_cache_hit_rate=/{print $2; exit}' "${summary_file}")"
  if [[ -z "${hit_rate}" ]]; then
    echo "Failed to parse prefill_cache_hit_rate from ${summary_file}" >&2
    return
  fi

  echo "Annotating JSONL files for policy ${policy} with prefill_cache_hit_rate=${hit_rate}"

  local f
  for f in "${RESULT_DIR}/${policy}_${DATASET}_"*rps.jsonl; do
    [[ -f "${f}" ]] || continue
    python3 - "$f" "$hit_rate" << 'EOF'
import json
import sys

path = sys.argv[1]
rate = float(sys.argv[2])

with open(path) as f:
    lines = [json.loads(l) for l in f if l.strip()]

for obj in lines:
    obj["prefill_cache_hit_rate"] = rate

with open(path, "w") as f:
    for obj in lines:
        f.write(json.dumps(obj) + "\n")
EOF
  done
}

cleanup() {
  stop_server
}
trap cleanup EXIT

for mem_fraction in "${MEM_FRACTION_LIST[@]}"; do
  mem_fraction="$(echo "${mem_fraction}" | xargs)"
  [[ -z "${mem_fraction}" ]] && continue

  RESULT_DIR="${BASE_RESULT_DIR}/mem_${mem_fraction}"
  mkdir -p "${RESULT_DIR}"

  echo "==== Running benchmarks with mem_fraction=${mem_fraction} (results in ${RESULT_DIR}) ===="

  for policy in tlru lru; do
    echo "==== Running ${policy} benchmark ===="
    start_server "${policy}" "${mem_fraction}"
    run_benchmark "${policy}" || true
    summarize_hit_rate "${policy}" || true
    add_hit_rate_to_jsonl_files "${policy}" || true
    stop_server
  done
done

echo "Benchmark results saved to ${BASE_RESULT_DIR}."