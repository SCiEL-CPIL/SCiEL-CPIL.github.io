"""Job-Shop dispatch via refined non-linear urgency-speed heuristic with smooth scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Refined non-linear urgency-speed product with:
    - Smoothed critical path via log-sum-exp
    - Multi-criteria refinement with tuned weights
    - Variability-aware speed adjustment
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
    
    # Smooth critical path via log-sum-exp (temperature-controlled soft-max)
    job_works = np.array(list(job_remaining_work.values()), dtype=np.float64)
    if len(job_works) > 0:
        temp = 5.0  # temperature for soft-max
        shifted = job_works / temp
        max_shifted = np.max(shifted)
        log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        current_cp = log_sum_exp * temp
    else:
        current_cp = 0.0
    
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
    
    # Enhanced critical path score with variability adjustment
    # Use a smooth surrogate: combine remaining work with spread penalty
    cp_score = remaining_works * (1.0 + 0.15 * duration_spread / (mean_durations + 1e-9))
    cp_norm = normalize(cp_score)
    
    # Non-linear urgency-speed product with refined formulation
    scores = np.zeros(n)
    for i, op in enumerate(ready_operations):
        job = op["job"]
        current_job_work = job_remaining_work.get(job, 0)
        
        if current_cp > 0:
            # Smoothed urgency: use a power law to emphasize high-urgency jobs
            urgency = (current_job_work / (current_cp + 1e-9)) ** 1.2
        else:
            urgency = 1.0
        
        # Speed with variability adjustment: penalize high variance slightly
        base_speed = 1.0 / (min_durations[i] + 1e-9)
        variability_penalty = 1.0 / (1.0 + 0.1 * duration_spread[i] / (mean_durations[i] + 1e-9))
        speed = base_speed * variability_penalty
        
        # Non-linear product with slight saturation to avoid extreme values
        raw_score = urgency * speed
        # Apply a mild compression to reduce sensitivity to outliers
        scores[i] = raw_score / (1.0 + 0.05 * raw_score)
    
    # Normalize the urgency-speed scores
    us_norm = normalize(scores)
    
    # Composite score with refined weights
    # Increase weight on the dominant non-linear signal, reduce others slightly
    final_scores = (
        us_norm * 0.65 +          # Non-linear urgency-speed (dominant signal, increased)
        cp_norm * 0.12 +          # Enhanced critical path (slightly reduced)
        spt_norm * 0.10 +         # Shortest processing time
        rem_ops_norm * 0.07 +     # Fewer remaining ops (slightly reduced)
        constraint_norm * 0.06    # Fewer machine options (slightly reduced)
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
