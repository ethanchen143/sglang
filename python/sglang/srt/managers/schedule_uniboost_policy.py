"""UniBoost scheduling policy for SGLang.

Port of Algorithm 1 + 2 from the paper appendix:
  - Boost(x, gamma) = (1/gamma) * log(1 / (1 - e^(-gamma*x)))
  - Quantize(s, k)  = k * 2^floor(log2(max(1, s/k)))       # MemGuard
  - Signal(r)       = prompt tokens (prefill) | attained decode tokens (decode)
  - pi(r)           = a_r - Boost(Quantize(Signal(r), k), gamma)
  - UpdateGamma     = EMA of (log(99)-log(95)) / (P99-P95), clipped
"""

from __future__ import annotations

from collections import deque
from math import exp, log, log2, floor
from typing import Deque, Optional

import numpy as np


# --- helper functions --------------------------
def boost(x: float, gamma: float) -> float:
    """b_gamma(x) = (1/gamma) * log(1 / (1 - e^(-gamma*x)))."""
    if x <= 0:
        x = 1e-6
    denom = 1.0 - exp(-gamma * x)
    if denom <= 1e-10:
        return x
    return (1.0 / gamma) * log(1.0 / denom)


def quantize(s: float, k: int) -> int:
    """Geometric bin anchored at k:  k * 2^floor(log2(max(1, s/k)))."""
    ratio = max(1.0, s / k)
    return int(k * (2 ** floor(log2(ratio))))


def signal(is_prefill: bool, prompt_tokens: int, decoded_tokens: int) -> int:
    """Prediction-free, phase-aware work signal.
    Prefill: prompt tokens (known).  Decode: attained decode tokens.
    """
    return prompt_tokens if is_prefill else decoded_tokens


def priority(arrival_time: float, is_prefill: bool,
             prompt_tokens: int, decoded_tokens: int,
             gamma: float, k: int) -> float:
    """pi(r) = a_r - Boost(Quantize(Signal(r), k), gamma). Smaller = served earlier."""
    s = signal(is_prefill, prompt_tokens, decoded_tokens)
    return arrival_time - boost(quantize(s, k), gamma)


# --- adaptive gamma (UpdateGamma) ----------------------------
class GammaAdaptiveTracker:
    """EMA-smoothed slope estimator, clipped to [gamma_min, gamma_max]."""

    def __init__(self,
                 initial_gamma: float = 3e-4,
                 beta: float = 0.3,              # EMA weight on new estimate
                 gamma_min: float = 1e-6,
                 gamma_max: float = 1.0,
                 eps: float = 1e-6,              # guards tail compression
                 update_interval: int = 50,
                 min_samples: int = 2000,
                 ring_size: int = 4000) -> None:
        self._gamma = float(initial_gamma)
        self._beta = beta
        self._gamma_min = gamma_min
        self._gamma_max = gamma_max
        self._eps = eps
        self._update_interval = update_interval
        self._min_samples = min_samples
        self._latencies: Deque[float] = deque(maxlen=ring_size)
        self._since_update = 0

    @property
    def gamma(self) -> float:
        return self._gamma

    def record(self, e2e_latency: float) -> None:
        if e2e_latency > 0:
            self._latencies.append(e2e_latency)
        self._since_update += 1

    def maybe_update(self) -> Optional[float]:
        if self._since_update < self._update_interval:
            return None
        if len(self._latencies) < self._min_samples:
            return None
        self._since_update = 0
        raw = self._estimate()
        if raw is None:
            return None
        # EMA smoothing, then clip to bounds.
        prev = self._gamma
        self._gamma = (1.0 - self._beta) * self._gamma + self._beta * raw
        self._gamma = min(max(self._gamma, self._gamma_min), self._gamma_max)
        n = len(self._latencies)
        print(f"[uniboost] gamma update: n={n} raw={raw:.6g} prev={prev:.6g} new={self._gamma:.6g}", flush=True)
        return self._gamma

    def _estimate(self) -> Optional[float]:
        n = len(self._latencies)
        if n < 50:
            return None
        idx95 = int(n * 0.95)
        idx99 = min(int(n * 0.99), n - 1)
        if idx99 <= idx95:
            return None
        arr = np.fromiter(self._latencies, dtype=np.float64, count=n)
        arr = np.partition(arr, [idx95, idx99])
        delta = max(arr[idx99] - arr[idx95], self._eps)      # guard against tail compression
        raw = (log(99.0) - log(95.0)) / delta
        return raw if raw > 0 else None


# --- module-level singleton shared by sort path + retract path ------------
_TRACKER: Optional[GammaAdaptiveTracker] = None


def init_tracker(initial_gamma: float, adaptive: bool,
                 beta: float, gamma_min: float, gamma_max: float,
                 update_interval: int, min_samples: int) -> None:
    global _TRACKER
    _TRACKER = GammaAdaptiveTracker(
        initial_gamma=initial_gamma,
        beta=beta,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        update_interval=update_interval,
        min_samples=min_samples,
    ) if adaptive else None


def current_gamma(default: float) -> float:
    return _TRACKER.gamma if _TRACKER is not None else default


def record_latency(e2e_latency: float) -> None:
    if _TRACKER is not None:
        _TRACKER.record(e2e_latency)
        _TRACKER.maybe_update()


def record_completion(req) -> None:
    """Pull e2e latency from req.time_stats and feed UpdateGamma.
    No-op when adaptive gamma is disabled. Safe to call from any finish path.
    """
    if _TRACKER is None:
        return
    ts = req.time_stats
    if ts is None or ts.completion_time <= 0 or ts.wait_queue_entry_time <= 0:
        return
    latency = ts.completion_time - ts.wait_queue_entry_time
    if latency > 0:
        _TRACKER.record(latency)
        _TRACKER.maybe_update()


# --- convenience helper for call sites in schedule_policy / schedule_batch ---
def priority_for_req(req, gamma: float, k: int) -> float:
    """Uniform priority computation for any Req object.
    A waiting req has decoded_tokens == 0; a running decode req has output_ids populated.
    """
    decoded = len(req.output_ids)
    is_prefill = decoded == 0                    # matches Signal() branch
    return priority(
        arrival_time=req.time_stats.wait_queue_entry_time,
        is_prefill=is_prefill,
        prompt_tokens=len(req.origin_input_ids),
        decoded_tokens=decoded,
        gamma=gamma,
        k=k,
    )