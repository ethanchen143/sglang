"""Build a 70% reasoning / 30% chat mixture for the UniBoost tail-latency bench.

Output JSONL is consumed by sglang.bench_serving via `--dataset-name custom`
(see python/sglang/benchmark/datasets/custom.py for the schema).

Paper mixture: 70% simplescaling/s1K (reasoning) + 30% ShareGPT (chat).

Usage:
    python build_dataset.py --out mix.jsonl --total 10000
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def load_s1k(n: int) -> list[tuple[str, str]]:
    ds = load_dataset("simplescaling/s1K", split="train")
    rows = []
    for r in ds:
        q = r.get("question") or r.get("prompt") or r.get("problem")
        a = (
            r.get("solution")
            or r.get("answer")
            or r.get("thinking_trajectories")
            or r.get("deepseek_thinking_trajectory")
        )
        if isinstance(a, list):
            a = "\n".join(map(str, a))
        if q and a:
            rows.append((str(q).strip(), str(a).strip()))
    random.shuffle(rows)
    return rows[:n]


def load_sharegpt(n: int) -> list[tuple[str, str]]:
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
    )
    rows = []
    for r in ds:
        conv = r.get("conversations") or []
        if len(conv) < 2:
            continue
        q = conv[0].get("value", "")
        a = conv[1].get("value", "")
        if q and a and conv[0].get("from") in (None, "human", "user"):
            rows.append((str(q).strip(), str(a).strip()))
        if len(rows) >= n * 3:
            break
    random.shuffle(rows)
    return rows[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--total", type=int, default=10_000)
    p.add_argument("--reason-frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed)
    n_reason = int(a.total * a.reason_frac)
    n_chat = a.total - n_reason

    reason = load_s1k(n_reason)
    chat = load_sharegpt(n_chat)
    rows = reason + chat
    random.shuffle(rows)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for q, r in rows:
            f.write(
                json.dumps(
                    {"conversations": [{"content": q}, {"content": r}]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(rows)} prompts -> {out}  ({len(reason)} reason / {len(chat)} chat)")


if __name__ == "__main__":
    main()
