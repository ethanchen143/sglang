## UniBoost tail-latency benchmark

- **Model:** Llama-3-8B, TP=1
- **Workload:** 70% `simplescaling/s1K` (reasoning) + 30% ShareGPT (chat), shuffled
- **Load:** Poisson arrivals at a target QPS (single value or sweep)
- **Metrics:** TTFT and end-to-end latency (mean, median, P95, P99) —
  computed by `sglang.bench_serving` directly

Everything reuses existing sglang infrastructure:

| What | sglang component |
|---|---|
| Request loader (JSONL → DatasetRow) | `python/sglang/benchmark/datasets/custom.py` |
| HTTP driver + percentile stats       | `python/sglang/bench_serving.py` |
| Server launch                        | `python/sglang/launch_server.py` |

The only thing this directory adds is the mixture builder.

### Prereqs

- GPU with ≥40GB VRAM for Llama-3-8B FP16 (paper uses A100 80GB)
- `pip install datasets` for the dataset pull
- `huggingface-cli login` for gated Llama weights
- sglang with the UniBoost patch applied (see `../../python/sglang/srt/managers/uniboost_policy.py`).
  The script auto-skips `uniboost` if the `--uniboost-gamma` flag isn't present.

### One-time: build the dataset

```bash
python build_dataset.py --out mix.jsonl --total 10000
```

### Run the sweep

```bash
./run_bench.sh
```

The driver launches sglang once per policy in `$POLICIES`, runs
`sglang.bench_serving` against each QPS in `$QPS`, and tears the server down
between policies. Per-request JSONLs + bench_serving's own summary land in
`results/<timestamp>/`.

### Common overrides

```bash
# Smoke test (5 min)
NUM_PROMPTS=500 QPS=0.25 ./run_bench.sh

# Fig. 5-style load sweep
QPS="0.20 0.24 0.26 0.28" NUM_PROMPTS=2000 ./run_bench.sh

# Add `lof` as a worst-case reference (longest-output-first)
POLICIES="fcfs lof uniboost" ./run_bench.sh
```

All env-var knobs:

| Var | Default |
|---|---|
| `MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| `TP`, `CHUNK_SIZE` | `1`, `1024` |
| `NUM_PROMPTS`, `WARMUP` | `10000`, `20` |
| `POLICIES`, `QPS` | `fcfs uniboost`, `0.25` |
| `UNIBOOST_GAMMA`, `UNIBOOST_K`, `UNIBOOST_BETA`, `UNIBOOST_ADAPTIVE` | `3e-4`, `128`, `0.3`, `1` |
