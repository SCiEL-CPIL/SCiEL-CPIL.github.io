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
    
    Strategy: Multi-criteria heuristic using soft-min for machine selection
    and composite scoring for operation priority.
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
    remaining_ops = np.zeros(n)
    
    # Use soft-min (log-sum-exp based) for smooth machine duration estimation
    # This gives a smoothed estimate of the best machine duration
    soft_min_durations = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        durations = op["machine_options"]
        remaining_works[i] = op["remaining_work"]
        remaining_ops[i] = op["remaining_ops"]
        
        if durations:
            dur_vals = np.array(list(durations.values()))
            # Soft-min: -log(sum(exp(-d/T))) * T, with T=1 for simplicity
            # This is a smooth approximation of min(durations)
            # Using the log-sum-exp trick for numerical stability
            neg_d = -dur_vals
            max_neg_d = np.max(neg_d)
            soft_min = -max_neg_d - np.log(np.sum(np.exp(neg_d - max_neg_d)))
            soft_min_durations[i] = soft_min
        else:
            soft_min_durations[i] = 0.0
    
    # Normalize features to [0, 1] range
    rw_min, rw_max = remaining_works.min(), remaining_works.max()
    md_min, md_max = soft_min_durations.min(), soft_min_durations.max()
    ro_min, ro_max = remaining_ops.min(), remaining_ops.max()
    
    rw_range = rw_max - rw_min if rw_max > rw_min else 1.0
    md_range = md_max - md_min if md_max > md_min else 1.0
    ro_range = ro_max - ro_min if ro_max > ro_min else 1.0
    
    rw_norm = (remaining_works - rw_min) / rw_range
    md_norm = (soft_min_durations - md_min) / md_range
    ro_norm = (remaining_ops - ro_min) / ro_range
    
    # Composite score: we want to minimize this
    # - Higher remaining_work -> prioritize (negative weight)
    # - Lower soft_min_duration -> prioritize (positive weight for duration)
    # - Lower remaining_ops -> prioritize (negative weight)
    
    # Adjusted weights: remaining work is most important, then remaining ops, then duration
    w_rw = 1.2
    w_ro = 0.4
    w_md = 0.3
    
    scores = -w_rw * rw_norm + w_md * md_norm - w_ro * ro_norm
    
    # Pick the operation with the lowest score (highest priority)
    idx = int(np.argmin(scores))
    return idx
# EVOLVE-BLOCK-END
