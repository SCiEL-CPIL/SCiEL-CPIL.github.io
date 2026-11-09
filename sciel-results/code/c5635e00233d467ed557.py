"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard: if we've spent too long, fall back to a fast heuristic.
    t_start = _time.perf_counter()
    deadline = t_start + 0.05  # 50ms safety margin per dispatch call

    # Precompute features for each ready operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    
    # Minimum processing time among machine options (bottleneck indicator)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Number of machine options (flexibility)
    n_options = np.array([
        len(op["machine_options"]) if op["machine_options"] else 1
        for op in ready_operations
    ], dtype=np.float64)
    
    # Remaining operations count (urgency: fewer remaining = closer to done)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Average duration across options
    avg_durations = np.array([
        (sum(op["machine_options"].values()) / len(op["machine_options"])) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)

    # Check deadline
    if _time.perf_counter() - t_start > deadline:
        # Fast fallback: max remaining work
        return int(np.argmax(remaining_works))

    # Composite scoring:
    # 1. Bottleneck elimination: prefer ops with large minimum duration (LPT)
    #    Normalize to [0,1]
    min_d_min, min_d_max = min_durations.min(), min_durations.max()
    if min_d_max > min_d_min:
        norm_min_d = (min_durations - min_d_min) / (min_d_max - min_d_min)
    else:
        norm_min_d = np.ones(n)

    # 2. Remaining work urgency
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    if rw_max > rw_min:
        norm_rw = (remaining_works - rw_min) / (rw_max - rw_min)
    else:
        norm_rw = np.ones(n)

    # 3. Flexibility: fewer options = less flexible = higher priority
    #    Invert so that fewer options -> higher score
    no_min, no_max = n_options.min(), n_options.max()
    if no_max > no_min:
        norm_flex = 1.0 - (n_options - no_min) / (no_max - no_min)
    else:
        norm_flex = np.ones(n)

    # 4. Urgency: fewer remaining ops = closer to completion = higher priority
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    if ro_max > ro_min:
        norm_urgency = 1.0 - (remaining_ops - ro_min) / (ro_max - ro_min)
    else:
        norm_urgency = np.ones(n)

    # Weighted composite score
    # Emphasize bottleneck elimination (LPT) and remaining work
    w_lpt = 0.35
    w_rw = 0.30
    w_flex = 0.20
    w_urgency = 0.15

    scores = (
        w_lpt * norm_min_d +
        w_rw * norm_rw +
        w_flex * norm_flex +
        w_urgency * norm_urgency
    )

    # Check deadline again
    if _time.perf_counter() - t_start > deadline:
        return int(np.argmax(scores))

    # Tie-breaking: among top candidates, prefer shorter average duration (SPT-like)
    # to reduce local machine idle time
    top_k = min(3, n)
    top_indices = np.argsort(scores)[-top_k:][::-1]  # descending by score
    
    if top_k == 1:
        return int(top_indices[0])
    
    # Among top candidates, pick the one with shortest avg duration
    best_idx = int(top_indices[0])
    best_avg = avg_durations[best_idx]
    for idx in top_indices[1:]:
        if avg_durations[idx] < best_avg:
            best_idx = int(idx)
            best_avg = avg_durations[idx]

    return best_idx
# EVOLVE-BLOCK-END
