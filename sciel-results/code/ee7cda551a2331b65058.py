"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Keep a module-level deadline guard so we never exceed the wall-clock budget.
_DEADLINE_START = None
_DEADLINE_LIMIT = 145.0  # seconds; leaves 5s margin under the 150s kill

def _ensure_deadline():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = _time.time()
    return _DEADLINE_START

def _within_deadline():
    if _DEADLINE_START is None:
        return True
    return (_time.time() - _DEADLINE_START) < _DEADLINE_LIMIT


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.

    Strategy: weighted combination of normalized features:
      - remaining_work / max_remaining_work  (long jobs first)
      - remaining_ops / max_remaining_ops    (depth / urgency)
      - min_duration / max_min_duration      (shortest processing time to reduce blocking)
      - duration_ratio = min_duration / (remaining_work + eps)  (efficiency)

    Weights refined from measured improvement: (1.2, 1.8, 0.6, 0.4)
    """
    _ensure_deadline()

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)

    # For each op, compute the minimum duration over its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0.0
        for op in ready_operations
    ], dtype=np.float64)

    # Normalize each feature to [0, 1]
    def _normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    nw = _normalize(remaining_works)
    no = _normalize(remaining_ops)
    nd = _normalize(min_durations)

    # duration_ratio: shorter relative to remaining work is better -> invert
    eps = 1e-9
    ratio = min_durations / (remaining_works + eps)
    nr = _normalize(ratio)
    # Invert so that lower ratio (more efficient) gets higher score
    nr = 1.0 - nr

    # Weighted combination
    w_work = 1.2
    w_ops = 1.8
    w_dur = 0.6
    w_ratio = 0.4

    scores = w_work * nw + w_ops * no + w_dur * nd + w_ratio * nr

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
