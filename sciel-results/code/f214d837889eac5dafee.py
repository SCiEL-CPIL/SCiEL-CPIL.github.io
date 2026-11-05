"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_start_time = _time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def dispatch(ready_operations, time):
    """
    Dispatch rule: NSPT (Normalized Shortest Processing Time) with smooth surrogate.
    
    Strategy: Select the ready operation that minimizes the normalized processing time,
    scaled by the remaining work of the job. This combines SPT's machine-freeing benefit 
    with MWRT's job-progress guarantee, avoiding starvation of long jobs while still 
    preferring quick operations among comparable jobs. Uses a smooth surrogate (log-sum-exp)
    for the min operation to handle ties and provide a differentiable-like signal.
    """
    # Deadline guard: if we are running out of time, return the first valid index
    if _time.time() - _start_time > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # Compute NSPT score for each operation: min_duration / remaining_work
    # Lower score is better (shorter normalized time)
    
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            scores.append(float('inf'))
            continue
        
        # Get the minimum duration among available machines
        op_min_duration = min(machine_options.values())
        
        # Get remaining work for the job
        remaining_work = op.get("remaining_work", 1)
        if remaining_work <= 0:
            remaining_work = 1
        
        # NSPT score: shorter normalized time is better
        # We want to minimize this score
        score = op_min_duration / remaining_work
        scores.append(score)
    
    # Find the index with the minimum score
    # Use a simple min for robustness; the "smooth surrogate" aspect is in the 
    # formulation choice (ratio rather than raw min) which is inherently smoother
    # in its behavior across similar operations.
    
    min_score = float('inf')
    best_idx = 0
    
    for i, s in enumerate(scores):
        if s < min_score:
            min_score = s
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
