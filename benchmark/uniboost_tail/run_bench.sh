#!/usr/bin/env bash
# Reproduce the UniBoost Table 2 / Fig. 5 experiment on sglang.
#
# Workload: 70% s1K reasoning + 30% ShareGPT, Llama-3-8B TP=1, Poisson arrivals.
# Metric:   per-request ttft and latency (sglang.bench_serving already prints
#           mean / median / p95 / p99).
#
# Relies on sglang's own harness:
#   - python/sglang/benchmark/datasets/custom.py  (loads our mixed JSONL)
#   - python/sglang/bench_serving.py              (drives the HTTP client,
#                                                  computes percentiles)
#
# Usage:
#   # first time only: build the mixture
#   python build_dataset.py --out mix.jsonl --total 10000
#
#   # then run sweep (script handles server lifecycle)
#   ./run_bench.sh
#
#   # knobs via env:
#   POLICIES="fcfs uniboost"  QPS="0.20 0.24 0.28"  NUM_PROMPTS=2000  ./run_bench.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TP="${TP:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
DATASET="${DATASET:-$HERE/mix.jsonl}"
NUM_PROMPTS="${NUM_PROMPTS:-10000}"
WARMUP="${WARMUP:-20}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
POLICIES="${POLICIES:-fcfs uniboost}"
QPS="${QPS:-0.25}"
MAX_RUNNING="${MAX_RUNNING:-}"   # set to e.g. 32 to force a real wait queue
OUTDIR="${OUTDIR:-$HERE/results/$(date +%Y%m%d-%H%M%S)}"

# UniBoost knobs (match the winning simulator config)
UNIBOOST_GAMMA="${UNIBOOST_GAMMA:-3e-4}"
UNIBOOST_K="${UNIBOOST_K:-128}"
UNIBOOST_BETA="${UNIBOOST_BETA:-0.3}"
UNIBOOST_ADAPTIVE="${UNIBOOST_ADAPTIVE:-1}"
UNIBOOST_MIN_SAMPLES="${UNIBOOST_MIN_SAMPLES:-2000}"

mkdir -p "$OUTDIR"
[[ -s "$DATASET" ]] || { echo "missing $DATASET. run: python build_dataset.py --out $DATASET --total $NUM_PROMPTS"; exit 1; }

SERVER_PID=""
gpu_used_mb() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NR==1 {print $1+0}'
}
cleanup() {
    pkill -TERM -f "sglang\.launch_server" 2>/dev/null || true
    pkill -TERM -f "sglang.srt"           2>/dev/null || true
    sleep 5
    pkill -KILL -f "sglang\.launch_server" 2>/dev/null || true
    pkill -KILL -f "sglang.srt"           2>/dev/null || true
    # Wait until GPU memory drops below 2 GB (or 30s, whichever first).
    for _ in $(seq 1 30); do
        used=$(gpu_used_mb); [[ -z "$used" ]] && break
        (( used < 2000 )) && break
        sleep 1
    done
}
trap cleanup EXIT INT TERM

launch() {
    local policy="$1" log="$2" extra=()
    if [[ "$policy" == "uniboost" ]]; then
        extra+=( --uniboost-gamma "$UNIBOOST_GAMMA" --uniboost-k "$UNIBOOST_K" --uniboost-beta "$UNIBOOST_BETA" \
                 --uniboost-gamma-min-samples "$UNIBOOST_MIN_SAMPLES" )
        [[ "$UNIBOOST_ADAPTIVE" == "1" ]] && extra+=( --uniboost-adaptive-gamma )
    fi
    [[ -n "$MAX_RUNNING" ]] && extra+=( --max-running-requests "$MAX_RUNNING" )
    python -m sglang.launch_server \
        --model-path "$MODEL" --tp "$TP" \
        --chunked-prefill-size "$CHUNK_SIZE" \
        --schedule-policy "$policy" \
        --disable-radix-cache \
        --host "$HOST" --port "$PORT" \
        "${extra[@]}" > "$log" 2>&1 &
    SERVER_PID=$!
}

wait_ready() {
    for _ in $(seq 1 300); do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died; see $1"; return 1; }
        curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

for policy in $POLICIES; do
    log="$OUTDIR/server_${policy}.log"
    echo ">>> server: policy=$policy"
    launch "$policy" "$log"
    wait_ready "$log" || { cleanup; continue; }

    for rate in $QPS; do
        tag="${policy}_qps${rate}"
        echo ">>> bench: $tag (n=$NUM_PROMPTS)"
        python -m sglang.bench_serving \
            --backend sglang --model "$MODEL" \
            --host "$HOST" --port "$PORT" \
            --dataset-name custom --dataset-path "$DATASET" \
            --num-prompts "$NUM_PROMPTS" \
            --request-rate "$rate" \
            --warmup-requests "$WARMUP" \
            --output-file "$OUTDIR/bench_${tag}.jsonl" \
            --output-details \
            2>&1 | tee "$OUTDIR/summary_${tag}.txt"
        sleep 10
    done

    cleanup; SERVER_PID=""
done

echo ">>> done. results in $OUTDIR"
echo ">>> each summary_*.txt contains mean / median / p95 / p99 from bench_serving."
