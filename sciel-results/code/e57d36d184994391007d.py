"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

# Simple time guard: if we ever exceed a fraction of the wall-clock budget,
# fall back to a fast heuristic immediately.
_DEADLINE_START = _time.monotonic()
_DEADLINE_LIMIT = 135.0  # seconds; evaluator kills at 150s, leave margin.


def _deadline_exceeded():
    return (_time.monotonic() - _DEADLINE_START) > _DEADLINE_LIMIT


def _fast_score(ready_operations):
    """Fast multi-criteria score used as fallback / base."""
    n = len(ready_operations)
    if n == 0:
        return 0
    # Normalize remaining_work and remaining_ops to [0,1]
    rw = [op["remaining_work"] for op in ready_operations]
    ro = [op["remaining_ops"] for op in ready_operations]
    rw_min, rw_max = min(rw), max(rw)
    ro_min, ro_max = min(ro), max(ro)
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    ro_range = ro_max - ro_min if ro_max > ro_min else 1.0

    # Min duration across machine options
    min_durs = []
    for op in ready_operations:
        opts = op["machine_options"]
        min_durs.append(min(opts.values()) if opts else float('inf'))
    md_min, md_max = min(min_durs), max(min_durs)
    md_range = md_max - md_min if md_max > md_min else 1.0

    # Weights: prioritize remaining work (longest job first), then fewer remaining ops (finish soon),
    # then shorter duration (reduce contention).
    w_rw, w_ro, w_md = 0.55, 0.25, 0.20

    best_score = -1e18
    best_idx = 0
    for i in range(n):
        s_rw = (rw[i] - rw_min) / rw_range
        s_ro = 1.0 - (ro[i] - ro_min) / ro_range  # fewer remaining ops is better
        s_md = 1.0 - (min_durs[i] - md_min) / md_range  # shorter is better
        score = w_rw * s_rw + w_ro * s_ro + w_md * s_md
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # If deadline is close, use the fast heuristic.
    if _deadline_exceeded():
        return _fast_score(ready_operations)

    n = len(ready_operations)
    if n == 1:
        return 0

    # Compute base features
    rw = [op["remaining_work"] for op in ready_operations]
    ro = [op["remaining_ops"] for op in ready_operations]
    rw_min, rw_max = min(rw), max(rw)
    ro_min, ro_max = min(ro), max(ro)
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    ro_range = ro_max - ro_min if ro_max > ro_min else 1.0

    min_durs = []
    for op in ready_operations:
        opts = op["machine_options"]
        min_durs.append(min(opts.values()) if opts else float('inf'))
    md_min, md_max = min(min_durs), max(min_durs)
    md_range = md_max - md_min if md_max > md_min else 1.0

    # Precompute normalized features
    norm_rw = [(rw[i] - rw_min) / rw_range for i in range(n)]
    norm_ro = [1.0 - (ro[i] - ro_min) / ro_range for i in range(n)]  # fewer ops better
    norm_md = [1.0 - (min_durs[i] - md_min) / md_range for i in range(n)]  # shorter better

    # Try a small set of weight combinations to find the best dispatch
    # Weights: (w_rw, w_ro, w_md)
    weight_sets = [
        (0.55, 0.25, 0.20),
        (0.60, 0.20, 0.20),
        (0.50, 0.30, 0.20),
        (0.65, 0.15, 0.20),
        (0.45, 0.35, 0.20),
        (0.70, 0.10, 0.20),
        (0.55, 0.15, 0.30),
        (0.40, 0.30, 0.30),
        (0.60, 0.25, 0.15),
        (0.50, 0.20, 0.30),
    ]

    # For each weight set, compute the best index
    best_idx = _fast_score(ready_operations)
    best_score = -1e18

    for w_rw, w_ro, w_md in weight_sets:
        for i in range(n):
            score = w_rw * norm_rw[i] + w_ro * norm_ro[i] + w_md * norm_md[i]
            if score > best_score:
                best_score = score
                best_idx = i

    return best_idx
# EVOLVE-BLOCK-END
