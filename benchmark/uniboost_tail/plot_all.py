"""Visualize every run under results/: end-to-end latency, gamma trajectory,
preempt history. One combined PNG.

Usage: python plot_all.py [results_dir]
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_bench_summary(path):
    """bench_serving's --output-file writes one summary JSON per run."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        # Take last record (in case of appends).
        records = [json.loads(l) for l in f if l.strip()]
    return records[-1] if records else None


def discover(root):
    """Return list of (run_name, qps, summary_dict_per_policy, gamma_events, preempt_events)."""
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        bench_files = sorted(glob.glob(os.path.join(d, "bench_*_qps*.jsonl")))
        per_policy = {}
        qps_seen = None
        for bf in bench_files:
            m = re.search(r"bench_([^_]+)_qps([\d.]+)\.jsonl$", os.path.basename(bf))
            if not m:
                continue
            policy, qps = m.group(1), float(m.group(2))
            qps_seen = qps
            summary = parse_bench_summary(bf)
            if summary is not None:
                per_policy[policy] = summary
        if not per_policy:
            continue
        gamma = load_jsonl(os.path.join(d, "gamma_uniboost.jsonl"))
        preempt = load_jsonl(os.path.join(d, "preempt_uniboost.jsonl"))
        runs.append({
            "name": name, "qps": qps_seen,
            "policies": per_policy,
            "gamma": gamma, "preempt": preempt,
        })
    return runs


def fmt_ms(v):
    return f"{v:.0f}" if v is not None else "-"


def print_summary(runs):
    print(f"{'run':<20} {'qps':>5} {'policy':<10} "
          f"{'mean_e2e':>9} {'p99_e2e':>9} {'mean_ttft':>10} {'p99_ttft':>9} "
          f"{'γ_end':>10} {'preempt':>7}")
    for r in runs:
        for policy, s in r["policies"].items():
            g_end = r["gamma"][-1]["new"] if r["gamma"] and policy == "uniboost" else None
            p_n = len(r["preempt"]) if policy == "uniboost" else 0
            print(f"{r['name']:<20} {r['qps']!s:>5} {policy:<10} "
                  f"{fmt_ms(s.get('mean_e2e_latency_ms')):>9} "
                  f"{fmt_ms(s.get('p99_e2e_latency_ms')):>9} "
                  f"{fmt_ms(s.get('mean_ttft_ms')):>10} "
                  f"{fmt_ms(s.get('p99_ttft_ms')):>9} "
                  f"{(f'{g_end:.3e}' if g_end is not None else '-'):>10} "
                  f"{p_n:>7}")


def plot(runs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_lat, ax_ttft, ax_gamma, ax_pre = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    # Order runs by QPS, then by name (so qps-16, qps-16-mem-0.4, qps-16-mem-0.6 group together)
    ordered = sorted(runs, key=lambda r: (r["qps"] or 0, r["name"]))
    labels = [r["name"] for r in ordered]
    x = list(range(len(labels)))

    fcfs_mean = [r["policies"].get("fcfs", {}).get("mean_e2e_latency_ms") for r in ordered]
    fcfs_p99 = [r["policies"].get("fcfs", {}).get("p99_e2e_latency_ms") for r in ordered]
    uni_mean = [r["policies"].get("uniboost", {}).get("mean_e2e_latency_ms") for r in ordered]
    uni_p99 = [r["policies"].get("uniboost", {}).get("p99_e2e_latency_ms") for r in ordered]

    width = 0.35
    ax_lat.bar([i - width/2 for i in x], fcfs_mean, width, label="fcfs mean", color="#4C72B0")
    ax_lat.bar([i + width/2 for i in x], uni_mean, width, label="uniboost mean", color="#DD8452")
    ax_lat.plot(x, fcfs_p99, "o--", color="#1F3B6E", label="fcfs p99")
    ax_lat.plot(x, uni_p99, "s--", color="#A0522D", label="uniboost p99")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_lat.set_ylabel("E2E latency (ms)")
    ax_lat.set_title("End-to-end latency: fcfs vs uniboost (bars=mean, lines=p99)")
    ax_lat.legend(fontsize=8)
    ax_lat.grid(True, axis="y", alpha=0.3)

    fcfs_ttft_mean = [r["policies"].get("fcfs", {}).get("mean_ttft_ms") for r in ordered]
    fcfs_ttft_p99 = [r["policies"].get("fcfs", {}).get("p99_ttft_ms") for r in ordered]
    uni_ttft_mean = [r["policies"].get("uniboost", {}).get("mean_ttft_ms") for r in ordered]
    uni_ttft_p99 = [r["policies"].get("uniboost", {}).get("p99_ttft_ms") for r in ordered]
    ax_ttft.bar([i - width/2 for i in x], fcfs_ttft_mean, width, label="fcfs mean", color="#4C72B0")
    ax_ttft.bar([i + width/2 for i in x], uni_ttft_mean, width, label="uniboost mean", color="#DD8452")
    ax_ttft.plot(x, fcfs_ttft_p99, "o--", color="#1F3B6E", label="fcfs p99")
    ax_ttft.plot(x, uni_ttft_p99, "s--", color="#A0522D", label="uniboost p99")
    ax_ttft.set_xticks(x)
    ax_ttft.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_ttft.set_ylabel("TTFT (ms)")
    ax_ttft.set_yscale("log")
    ax_ttft.set_title("TTFT: fcfs vs uniboost (bars=mean, lines=p99)")
    ax_ttft.legend(fontsize=8)
    ax_ttft.grid(True, axis="y", which="both", alpha=0.3)

    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(ordered):
        if not r["gamma"]:
            continue
        t0 = r["gamma"][0]["t"]
        xs = [e["t"] - t0 for e in r["gamma"]]
        ax_gamma.plot(xs, [e["new"] for e in r["gamma"]], "-", lw=1.4,
                      color=cmap(i % 10), label=r["name"])
    ax_gamma.set_yscale("log")
    ax_gamma.set_xlabel("seconds since first update")
    ax_gamma.set_ylabel("gamma (EMA)")
    ax_gamma.set_title("Adaptive-gamma trajectories (uniboost runs)")
    ax_gamma.grid(True, which="both", alpha=0.3)
    ax_gamma.legend(fontsize=7, ncol=2)

    preempt_counts = [len(r["preempt"]) for r in ordered]
    ax_pre.bar(x, preempt_counts, color="#C44E52")
    ax_pre.set_xticks(x)
    ax_pre.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_pre.set_ylabel("# preempt events logged")
    ax_pre.set_title("Preemption events (uniboost)")
    ax_pre.grid(True, axis="y", alpha=0.3)
    if max(preempt_counts) == 0:
        ax_pre.text(0.5, 0.5, "0 preempt events across all runs\n(no KV-full retracts; priority preemption disabled)",
                    transform=ax_pre.transAxes, ha="center", va="center",
                    fontsize=11, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    runs = discover(root)
    if not runs:
        sys.exit(f"no runs found under {root}")
    print_summary(runs)
    plot(runs, os.path.join(root, "all_runs.png"))


if __name__ == "__main__":
    main()
