import json
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt


BASE_DIR = Path(__file__).resolve().parent / "benchmark_results_1500"


def parse_hit_rate_txt(path: Path) -> float | None:
    if not path.is_file():
        return None
    line = path.read_text().strip()
    if not line:
        return None
    # format: prefill_cache_hit_rate=0.141349  cached_tokens=... total_tokens=...
    parts = line.split()
    for p in parts:
        if p.startswith("prefill_cache_hit_rate="):
            try:
                return float(p.split("=", 1)[1])
            except ValueError:
                return None
    return None


def parse_jsonl_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


@st.cache_data
def load_data() -> pd.DataFrame:
    rows = []
    if not BASE_DIR.is_dir():
        return pd.DataFrame()

    for mem_dir in sorted(BASE_DIR.glob("mem_*")):
        mem_level = mem_dir.name.split("_", 1)[1] if "_" in mem_dir.name else mem_dir.name
        for policy in ["lru", "tlru"]:
            # hit rate from text file
            hit_rate_path = mem_dir / f"{policy}_hit_rate.txt"
            hit_rate = parse_hit_rate_txt(hit_rate_path)

            # summary metrics from jsonl files per RPS
            for json_file in sorted(mem_dir.glob(f"{policy}_sharegpt_*rps.jsonl")):
                metrics = parse_jsonl_summary(json_file)
                if not metrics:
                    continue

                rps = float(metrics.get("request_rate", 0.0))
                row = {
                    "mem_level": mem_level,
                    "policy": policy,
                    "rps": rps,
                    "hit_rate": hit_rate,
                    "mean_ttft_ms": metrics.get("mean_ttft_ms"),
                    "p99_ttft_ms": metrics.get("p99_ttft_ms"),
                    "mean_tpot_ms": metrics.get("mean_tpot_ms"),
                    "p99_tpot_ms": metrics.get("p99_tpot_ms"),
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["policy", "mem_level", "rps"])
    return df


def main():
    st.title("Benchmark: Cache + Latency Visualization")

    df = load_data()
    if df.empty:
        st.error(f"No data found under {BASE_DIR}")
        return

    st.sidebar.header("Controls")

    # Choose which policies to display (e.g., LRU vs TLRU)
    policies = sorted(df["policy"].unique().tolist())
    selected_policies = st.sidebar.multiselect(
        "Policies", policies, default=policies
    )

    # Choose metric
    metric_options = {
        "hit_rate": "Prefill cache hit rate",
        "mean_ttft_ms": "Mean TTFT (ms)",
        "p99_ttft_ms": "P99 TTFT (ms)",
        "mean_tpot_ms": "Mean TPoT (ms)",
        "p99_tpot_ms": "P99 TPoT (ms)",
    }
    metric_key = st.sidebar.selectbox(
        "Metric",
        list(metric_options.keys()),
        format_func=lambda k: metric_options[k],
    )

    # Optional: select which memory levels to show
    mem_levels = sorted(df["mem_level"].unique().tolist())
    selected_mem_levels = st.sidebar.multiselect(
        "Memory pool sizes to show (legend)", mem_levels, default=mem_levels
    )

    plot_df = df[
        (df["policy"].isin(selected_policies))
        & (df["mem_level"].isin(selected_mem_levels))
    ].copy()

    st.subheader(f"{metric_options[metric_key]} vs RPS")

    if plot_df.empty:
        st.warning("No data for this selection.")
        return

    chart = (
        alt.Chart(plot_df)
        .mark_point(size=150, filled=True)
        .encode(
            x=alt.X("rps:Q", title="Requests per second"),
            y=alt.Y(f"{metric_key}:Q", title=metric_options[metric_key]),
            color=alt.Color(
                "policy:N",
                title="Policy",
                scale=alt.Scale(
                    domain=["lru", "tlru"],
                    range=["#1f77b4", "#ff7f0e"],  # blue and orange
                ),
            ),
            shape=alt.Shape("mem_level:N", title="Memory pool size"),
            tooltip=[
                "policy",
                "mem_level",
                "rps",
                alt.Tooltip(metric_key, title=metric_options[metric_key]),
            ],
        )
        .properties(height=550)
    )

    st.altair_chart(chart, use_container_width=True)

    st.subheader("Raw data")
    st.dataframe(plot_df, use_container_width=True)


if __name__ == "__main__":
    main()