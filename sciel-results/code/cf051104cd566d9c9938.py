"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Uses a composite heuristic based on:
    - Critical Path Method (remaining work)
    - Minimum processing time (SPT) to reduce machine blocking
    - Constrained operations first (fewer machine options)
    - Jobs closer to completion (fewer remaining ops)
    - Load balancing: prefer machines that are less busy (approximated by shorter durations)
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    # Precompute features for all ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    min_durations = np.array([min(op["machine_options"].values()) if op["machine_options"] else 1.0 for op in ready_operations], dtype=np.float64)
    num_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Normalize features to [0, 1] range for better weighting
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    nw_norm = normalize(remaining_works)
    md_norm = normalize(min_durations)
    no_norm = normalize(num_options)
    ro_norm = normalize(remaining_ops)
    
    # Composite score:
    # - High remaining work is good (priority to critical path) - weight 1.0
    # - Shorter min duration is good (frees machine sooner) - weight 0.3
    # - Fewer machine options is better (constrained operations first) - weight 0.15
    # - Fewer remaining ops is better (closer to completion) - weight 0.1
    
    scores = (
        nw_norm * 1.0 +
        (1.0 - md_norm) * 0.3 +
        (1.0 - no_norm) * 0.15 +
        (1.0 - ro_norm) * 0.1
    )
    
    # Tie-breaking: if scores are very close, prefer the one with shorter min_duration
    max_score = np.max(scores)
    close_to_max = np.where(scores > max_score - 1e-9)[0]
    if len(close_to_max) > 1:
        # Among tied scores, pick the one with shortest min_duration
        tied_durations = min_durations[close_to_max]
        best_tie_idx = close_to_max[np.argmin(tied_durations)]
        return int(best_tie_idx)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
