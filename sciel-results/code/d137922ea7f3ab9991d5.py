"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import numpy as np

# Deadline guard: evaluator kills after 150s wall-clock; keep a safety margin.
_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _fast_dispatch(ready_operations, time):
    """
    Fast heuristic dispatch rule for job-shop scheduling.
    
    Strategy: Multi-criteria scoring combining:
    1. Remaining work (longest job remaining) - primary
    2. Number of remaining operations (urgency)
    3. Minimum processing time on available machines (shortest job next)
    4. Machine load balancing consideration
    
    Returns index of best ready operation to start.
    """
    n = len(remaining_ops_list := [op["remaining_ops"] for op in ready_operations])
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops_counts = [op["remaining_ops"] for op in ready_operations]
    
    # Compute minimum duration for each operation (best case processing time)
    min_durs = []
    for op in ready_operations:
        opts = op["machine_options"]
        if isinstance(opts, dict):
            min_durs.append(min(opts.values()) if opts else 0)
        elif isinstance(opts, (list, tuple, np.ndarray)):
            min_durs.append(min(opts) if len(opts) > 0 else 0)
        else:
            min_durs.append(0)
    
    # Normalize each criterion
    def normalize(arr):
        arr = np.array(arr, dtype=np.float64)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    # Criterion 1: More remaining work is better (priority to long jobs)
    c1 = normalize(remaining_works)
    
    # Criterion 2: More remaining operations means the job has more to go, 
    # but we want to finish jobs - so fewer remaining ops might be preferred 
    # to complete jobs quickly. Actually, let's think: if a job has many remaining ops,
    # it's a long job. We want to prioritize jobs that are close to completion 
    # to reduce makespan? Or prioritize long jobs to avoid blocking?
    # In job-shop, a common good heuristic is to prioritize jobs with more 
    # remaining work (to avoid idle machines waiting for short jobs to finish).
    # Let's use: more remaining ops = higher priority (longer jobs first)
    c2 = normalize(remaining_ops_counts)
    
    # Criterion 3: Shorter immediate processing time (to keep machines busy)
    c3 = 1.0 - normalize(min_durs)  # Invert: shorter is better
    
    # Weighted combination
    weights = np.array([0.4, 0.3, 0.3], dtype=np.float64)
    scores = weights[0] * c1 + weights[1] * c2 + weights[2] * c3
    
    # Add small random tiebreaker to avoid deterministic pitfalls
    scores += _random.random() * 1e-6
    
    return int(np.argmax(scores))

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Args:
        ready_operations: list of dicts, each with keys:
            - job: job identifier
            - op: operation identifier  
            - machine_options: dict {machine: duration} or similar
            - remaining_ops: number of remaining operations in the job
            - remaining_work: total remaining processing time for the job
        time: current simulation time
    
    Returns:
        int: index of the ready operation to start next
    """
    if _deadline_exceeded():
        # Return a safe default
        return 0
    
    if not ready_operations:
        return 0
    
    return _fast_dispatch(ready_operations, time)
# EVOLVE-BLOCK-END
