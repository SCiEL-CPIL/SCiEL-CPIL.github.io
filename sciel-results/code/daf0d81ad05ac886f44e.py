"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # safety margin before 150s kill

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _soft_min(arr, temperature=0.1):
    """Soft-min using log-sum-exp for smoothness."""
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    a = -arr / temperature
    max_a = a.max()
    if np.isinf(max_a):
        return -max_a * temperature
    sum_exp = np.sum(np.exp(a - max_a))
    return -(max_a + np.log(sum_exp)) * temperature

def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    if _deadline_exceeded():
        # Fallback to simple SPT
        min_durations = np.array([
            min(op["machine_options"].values()) if op["machine_options"] else 1.0
            for op in ready_operations
        ], dtype=np.float64)
        return int(np.argmin(min_durations))

    # Greedy Lookahead Simulation
    # For each ready operation, simulate its completion and estimate the resulting makespan
    # We approximate the makespan by looking at the maximum remaining work among other jobs
    # that would be affected by this choice.
    
    # Features for each operation
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)
    
    # Normalize for consistent scaling
    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    nw = _norm(remaining_work)
    no = _norm(remaining_ops)
    nd = _norm(min_durations)
    
    # Lookahead: For each candidate i, estimate the "cost" of starting it now.
    # Cost = duration_i + max(remaining_work of other ops / some factor)
    # Actually, a better heuristic: 
    # If we start op i, we free up its machine after duration_i.
    # Other ops on the same machine are delayed.
    # But we don't know machine loads directly.
    
    # Alternative lookahead: 
    # The "projected makespan" contribution is roughly:
    # current_time + duration_i + max(remaining_work of all ops / n_machines)
    # This is constant for all i except duration_i.
    
    # Let's use a more nuanced lookahead:
    # For each i, score_i = duration_i * (1 + sum(remaining_work_j for j != i) / (n * avg_duration))
    # This penalizes long operations when there is a lot of remaining work.
    
    # Or, use the "critical path" idea:
    # Priority = remaining_work / duration (WSPT)
    # But we want to minimize makespan, so maybe we want to do the operation that has the highest "urgency" 
    # relative to its duration.
    
    # Let's try a hybrid:
    # 1. Base score: WSPT (remaining_work / min_duration)
    # 2. Lookahead adjustment: penalize if this op is long and there are many other ops waiting.
    
    # Compute WSPT
    with np.errstate(divide='ignore', invalid='ignore'):
        wspt = np.where(min_durations > 0, remaining_work / min_durations, 0.0)
    
    # Normalize WSPT
    wspt_norm = _norm(wspt)
    
    # Lookahead: For each i, the "delay" imposed on others is roughly duration_i.
    # If we pick a long op, we delay others. 
    # So we penalize long ops when the total remaining work is high.
    total_work = remaining_work.sum()
    avg_work = total_work / n
    
    # Penalty = min_duration_i * (total_work / avg_work) 
    # This is just min_duration_i * n, which is constant scaling.
    
    # Let's try: score = wspt_norm - w * nd
    # Where nd is normalized duration (higher duration = higher penalty)
    
    w_lookahead = 0.5
    score = wspt_norm - w_lookahead * nd
    
    # Also consider remaining_ops: jobs with more remaining ops are more "critical"
    # so we add a bonus for higher remaining_ops
    w_ops = 0.3
    score += w_ops * no
    
    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
