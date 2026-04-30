"""Lightweight JSONL trace writer for UniBoost benchmarking.

Enabled when env var UNIBOOST_TRACE_DIR is set. Writes two files into that
directory, optionally suffixed with UNIBOOST_TRACE_TAG:
  - gamma_<tag>.jsonl    one line per adaptive-gamma EMA update
  - preempt_<tag>.jsonl  one line per retract / priority preemption event
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional, TextIO


_LOCK = threading.Lock()
_GAMMA_FH: Optional[TextIO] = None
_PREEMPT_FH: Optional[TextIO] = None
_INITIALIZED = False


def _open(kind: str) -> Optional[TextIO]:
    out_dir = os.environ.get("UNIBOOST_TRACE_DIR")
    if not out_dir:
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        return None
    tag = os.environ.get("UNIBOOST_TRACE_TAG", "run")
    path = os.path.join(out_dir, f"{kind}_{tag}.jsonl")
    return open(path, "a", buffering=1)


def _ensure_init() -> None:
    global _GAMMA_FH, _PREEMPT_FH, _INITIALIZED
    if _INITIALIZED:
        return
    with _LOCK:
        if _INITIALIZED:
            return
        _GAMMA_FH = _open("gamma")
        _PREEMPT_FH = _open("preempt")
        _INITIALIZED = True


def _write(fh: Optional[TextIO], record: dict) -> None:
    if fh is None:
        return
    record.setdefault("t", time.time())
    try:
        with _LOCK:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except (OSError, ValueError):
        pass


def log_gamma_update(n: int, raw: float, prev: float, new: float) -> None:
    _ensure_init()
    _write(_GAMMA_FH, {"event": "gamma_update", "n": n, "raw": raw, "prev": prev, "new": new})


def log_preempt(kind: str, **fields) -> None:
    """kind: 'kv_full' | 'priority' | 'test'."""
    _ensure_init()
    record = {"event": "preempt", "kind": kind}
    record.update(fields)
    _write(_PREEMPT_FH, record)
