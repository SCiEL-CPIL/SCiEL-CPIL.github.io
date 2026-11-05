"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Combines:
    - Non-linear urgency-speed product (from look-ahead CP reduction)
    - Multi-criteria weighted scoring (MWR, SPT, remaining ops, constraint)
    - Enhanced critical path with processing time variability
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        return 0
    
    # Precompute features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    min_durations = np.array([min(op["machine_options"].values()) if op["machine_options"] else 1.0 for op in ready_operations], dtype=np.float64)
    mean_durations = np.array([np.mean(list(op["machine_options"].values())) if op["machine_options"] else 1.0 for op in ready_operations], dtype=np.float64)
    duration_spread = np.array([np.std(list(op["machine_options"].values())) if len(op["machine_options"]) > 1 else 0.0 for op in ready_operations], dtype=np.float64)
    num_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Compute job-level remaining work for urgency calculation
    job_remaining_work = {}
    for op in ready_operations:
        job = op["job"]
        if job not in job_remaining_work:
            job_remaining_work[job] = op["remaining_work"]
        else:
            job_remaining_work[job] = max(job_remaining_work[job], op["remaining_work"])
    
    current_cp = max(job_remaining_work.values()) if job_remaining_work else 0
    
    # Normalize features to [0, 1] range
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    mwr_norm = normalize(remaining_works)
    spt_norm = 1.0 - normalize(min_durations)
    rem_ops_norm = 1.0 - normalize(remaining_ops)
    constraint_norm = 1.0 - normalize(num_options)
    
    # Enhanced critical path score with variability penalty
    cp_score = remaining_works * (1.0 + 0.1 * duration_spread / (mean_durations + 1e-9))
    cp_norm = normalize(cp_score)
    
    # Non-linear urgency-speed product (key insight from inspiration 1)
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        job = op["job"]
        current_job_work = job_remaining_work.get(job, 0)
        
        if current_cp > 0:
            urgency = current_job_work / (current_cp + 1e-9)
        else:
            urgency = 1.0
        
        speed = 1.0 / (min_durations[i] + 1e-9)
        
        # Non-linear product: high urgency AND high speed = high priority
        scores[i] = urgency * speed
    
    # Normalize the urgency-speed scores
    us_norm = normalize(scores)
    
    # Composite score combining non-linear product with multi-criteria weights
    # Weight the non-linear product heavily (it captures the key insight)
    # Add multi-criteria as refinement
    final_scores = (
        us_norm * 0.6 +          # Non-linear urgency-speed (dominant signal)
        cp_norm * 0.15 +         # Enhanced critical path
        spt_norm * 0.1 +         # Shortest processing time
        rem_ops_norm * 0.08 +    # Fewer remaining ops
        constraint_norm * 0.07   # Fewer machine options
    )
    
    # Tie-breaking: prefer shorter min_duration among close scores
    max_score = np.max(final_scores)
    close_to_max = np.where(final_scores > max_score - 1e-9)[0]
    if len(close_to_max) > 1:
        tied_durations = min_durations[close_to_max]
        best_tie_idx = close_to_max[np.argmin(tied_durations)]
        return int(best_tie_idx)
    
    return int(np.argmax(final_scores))
# EVOLVE-BLOCK-END
