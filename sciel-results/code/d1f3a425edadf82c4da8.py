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
    - Critical Path: remaining work
    - Constraint: fewer machine options
    - SPT: shorter processing time
    - Urgency: fewer remaining operations
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
    
    # Handle division by zero
    min_durations_safe = np.where(min_durations > 0, min_durations, 1.0)
    num_options_safe = np.where(num_options > 0, num_options, 1.0)
    remaining_ops_safe = np.where(remaining_ops > 0, remaining_ops, 1.0)
    
    # Features
    # 1. Criticality: Higher remaining work is better
    criticality = remaining_works
    
    # 2. SPT: Shorter duration is better
    spt = 1.0 / min_durations_safe
    
    # 3. Constraint: Fewer options is better
    constraint = 1.0 / num_options_safe
    
    # 4. Urgency: Fewer remaining ops is better (job is closer to done)
    urgency = 1.0 / remaining_ops_safe
    
    # Normalize features to [0, 1]
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-12:
            return np.ones_like(arr) # If all same, give equal priority
        return (arr - min_val) / (max_val - min_val)
    
    crit_norm = normalize(criticality)
    spt_norm = normalize(spt)
    cons_norm = normalize(constraint)
    urg_norm = normalize(urgency)
    
    # Composite Score
    # Using a product emphasizes that an operation must be good in multiple aspects.
    # However, pure product can be too harsh. Let's use a weighted sum with tuned weights
    # based on typical JSS findings: Criticality is most important, then Constraint, then SPT.
    
    # Let's try a geometric mean (average of logs) which is equivalent to product of normalized scores
    # Score = exp(mean(log(crit_norm + eps), log(spt_norm + eps), ...))
    # This is robust.
    
    eps = 1e-9
    log_scores = (
        np.log(crit_norm + eps) +
        np.log(spt_norm + eps) +
        np.log(cons_norm + eps) +
        np.log(urg_norm + eps)
    )
    
    scores = np.exp(log_scores)
    
    # Tie-breaking: if scores are very close, prefer the one with shorter min_duration
    max_score = np.max(scores)
    close_to_max = np.where(scores > max_score - 1e-9)[0]
    if len(close_to_max) > 1:
        tied_durations = min_durations[close_to_max]
        best_tie_idx = close_to_max[np.argmin(tied_durations)]
        return int(best_tie_idx)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
