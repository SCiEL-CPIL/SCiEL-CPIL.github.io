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
    - Enhanced Critical Path: remaining work weighted by processing time variability
    - Minimum processing time (SPT) to reduce machine blocking
    - Constrained operations first (fewer machine options)
    - Jobs closer to completion (fewer remaining ops)
    - Machine load balancing: prefer operations that can run on less-contended machines
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    # Precompute features for all ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    
    # For each operation, compute min duration, mean duration, and spread
    min_durations = np.array([min(op["machine_options"].values()) if op["machine_options"] else 1.0 for op in ready_operations], dtype=np.float64)
    mean_durations = np.array([np.mean(list(op["machine_options"].values())) if op["machine_options"] else 1.0 for op in ready_operations], dtype=np.float64)
    duration_spread = np.array([np.std(list(op["machine_options"].values())) if len(op["machine_options"]) > 1 else 0.0 for op in ready_operations], dtype=np.float64)
    
    num_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Enhanced critical path: combine remaining work with a penalty for high variance
    # Operations with high variance in processing times are riskier
    cp_score = remaining_works * (1.0 + 0.1 * duration_spread / (mean_durations + 1e-9))
    
    # Normalize features to [0, 1] range for better weighting
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    cp_norm = normalize(cp_score)
    md_norm = normalize(min_durations)
    no_norm = normalize(num_options)
    ro_norm = normalize(remaining_ops)
    
    # Composite score:
    # - High enhanced critical path is good (priority to true critical path) - weight 1.0
    # - Shorter min duration is good (frees machine sooner) - weight 0.25
    # - Fewer machine options is better (constrained operations first) - weight 0.18
    # - Fewer remaining ops is better (closer to completion) - weight 0.07
    
    scores = (
        cp_norm * 1.0 +
        (1.0 - md_norm) * 0.25 +
        (1.0 - no_norm) * 0.18 +
        (1.0 - ro_norm) * 0.07
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
