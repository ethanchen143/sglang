"""Build a synthetic log-normal workload for the UniBoost tail-latency bench.

Both prompt and output lengths are drawn from independent log-normal
distributions. Each prompt/completion is materialized as random vocabulary
tokens decoded back to text, so when sglang's `custom` dataset loader (and the
server) re-tokenize the strings, the resulting token counts match the sampled
lengths within rounding.

Output JSONL is consumed by sglang.bench_serving via `--dataset-name custom`
(see python/sglang/benchmark/datasets/custom.py for the schema).

Why a fixed log-normal instead of s1K + ShareGPT?
    A closed-form distribution makes the offered load and tail behavior
    analytically tractable, which makes UniBoost's gamma / threshold
    interactions easier to reason about.

Usage:
    python build_dataset.py --out mix.jsonl --total 10000 \
        --prompt-median 256 --prompt-sigma 0.8 \
        --output-median 512 --output-sigma 1.2
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def sample_lognormal_lens(
    median: float, sigma: float, num: int, lo: int, hi: int, rng: np.random.Generator
) -> np.ndarray:
    """Log-normal lengths with median=`median`, shape=`sigma`, clipped to [lo, hi]."""
    mu = math.log(median)
    lens = rng.lognormal(mean=mu, sigma=sigma, size=num)
    return np.clip(np.round(lens), lo, hi).astype(int)


def make_text(tokenizer, n_tokens: int, rng: np.random.Generator) -> str:
    ids = rng.integers(0, tokenizer.vocab_size, size=n_tokens, dtype=np.int64).tolist()
    return tokenizer.decode(ids)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--total", type=int, default=10_000)
    p.add_argument(
        "--tokenizer",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HF model id used to materialize prompts/completions of a target token length",
    )
    p.add_argument("--prompt-median", type=float, default=256.0)
    p.add_argument("--prompt-sigma", type=float, default=0.8)
    p.add_argument("--prompt-min", type=int, default=8)
    p.add_argument("--prompt-max", type=int, default=4096)
    p.add_argument("--output-median", type=float, default=512.0)
    p.add_argument("--output-sigma", type=float, default=1.2)
    p.add_argument("--output-min", type=int, default=8)
    p.add_argument("--output-max", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    prompt_lens = sample_lognormal_lens(
        a.prompt_median, a.prompt_sigma, a.total, a.prompt_min, a.prompt_max, rng
    )
    output_lens = sample_lognormal_lens(
        a.output_median, a.output_sigma, a.total, a.output_min, a.output_max, rng
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"removing existing dataset {out}")
        out.unlink()
    with out.open("w") as f:
        for plen, olen in zip(prompt_lens, output_lens):
            f.write(
                json.dumps(
                    {
                        "conversations": [
                            {"content": make_text(tok, int(plen), rng)},
                            {"content": make_text(tok, int(olen), rng)},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def stats(x: np.ndarray) -> str:
        return (
            f"mean={x.mean():.0f} median={np.median(x):.0f} "
            f"p95={np.percentile(x, 95):.0f} p99={np.percentile(x, 99):.0f} "
            f"max={x.max()}"
        )

    print(f"wrote {a.total} prompts -> {out}")
    print(f"  prompt_len: {stats(prompt_lens)}")
    print(f"  output_len: {stats(output_lens)}")


if __name__ == "__main__":
    main()
