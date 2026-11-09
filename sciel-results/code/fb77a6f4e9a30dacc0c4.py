"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Module-level state for deadline guard
_START_TIME = None
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0  # seconds


def _check_deadline():
    """Check if we're approaching the deadline."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME > (_DEADLINE - _SAFETY_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Multi-criteria heuristic combining:
      1. Most remaining work (prioritize long jobs)
      2. Earliest completion time on best machine (shorter duration preferred)
      3. Least remaining operations (finish jobs soon to reduce makespan)
    
    We compute a composite score for each ready operation and pick the one
    with the highest priority (lowest score).
    """
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Deadline guard: if we're running low on time, use a simple fast heuristic
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Compute features for each ready operation
    remaining_works = np.zeros(n)
    min_durations = np.zeros(n)
    remaining_ops = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        durations = op["machine_options"]
        min_dur = min(durations.values()) if durations else 0.0
        min_durations[i] = min_dur
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]
    
    # Normalize features to [0, 1] range for robust weighting
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    md_min, md_max = min_durations.min(), min_durations.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    md_range = md_max - md_min if md_max > md_min else 1.0
    ro_range = ro_max - ro_min if ro_max > ro_min else 1.0
    
    rw_norm = (remaining_works - rw_min) / rw_range
    md_norm = (min_durations - md_min) / md_range
    ro_norm = (remaining_ops - ro_min) / ro_range
    
    # Composite score: we want to minimize this
    # - Higher remaining_work -> should be prioritized (negative weight)
    # - Lower min_duration -> should be prioritized (positive weight for duration)
    # - Lower remaining_ops -> should be prioritized (negative weight)
    
    # Weights: remaining work is most important, then remaining ops, then duration
    w_rw = 1.0
    w_ro = 0.3
    w_md = 0.2
    
    scores = -w_rw * rw_norm + w_md * md_norm - w_ro * ro_norm
    
    # Pick the operation with the lowest score (highest priority)
    idx = int(np.argmin(scores))
    return idx
# EVOLVE-BLOCK-END
