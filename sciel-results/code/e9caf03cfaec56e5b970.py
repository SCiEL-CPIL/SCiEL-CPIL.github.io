"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START = _time.monotonic()
_DEADLINE = _START + 145.0  # 5s safety margin

def _deadline_exceeded():
    return _time.monotonic() > _DEADLINE

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)

def _soft_min(values, alpha=2.0):
    """Compute soft-min using log-sum-exp trick for smooth approximation."""
    values = np.asarray(values, dtype=np.float64)
    neg_vals = -values
    max_neg = np.max(neg_vals)
    shifted = neg_vals - max_neg
    log_sum = max_neg + np.log(np.sum(np.exp(shifted)))
    return np.exp(-log_sum)

def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling with greedy lookahead.
    Returns index of the ready operation to start next.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Check deadline
    if _deadline_exceeded():
        return 0

    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # For each op, find the minimum duration among its machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)

    # Also compute max duration to capture range
    max_durations = np.array([
        max(op["machine_options"].values()) if op["machine_options"] else 1.0
        for op in ready_operations
    ], dtype=np.float64)

    # Normalize features
    nw = _normalize(remaining_work)
    no = _normalize(remaining_ops)
    nd_min = _normalize(min_durations)

    # Compute average work per remaining op
    avg_work_per_op = remaining_work / np.maximum(remaining_ops, 1.0)
    naw = _normalize(avg_work_per_op)

    # Greedy Lookahead Simulation:
    # For each candidate operation i, estimate the projected makespan if we start it now.
    # The idea: after starting op i, the remaining ready set will have their operations
    # delayed. We estimate the maximum completion time across all jobs.
    
    # Projected completion time for each job if we DON'T start it yet:
    # Each remaining operation j will take at least min_durations[j] time.
    # The total remaining work for job j's remaining ops is remaining_work[j].
    # A simple estimate: the job's remaining work will be completed in roughly
    # remaining_work[j] / (number of machines available for that job).
    # But we don't have machine availability info, so use a simpler model.
    
    # Model: If we start op i now, it takes min_durations[i] time.
    # During that time, other operations can't progress (simplified).
    # After op i completes, the remaining operations' completion times are
    # estimated as: time + min_durations[i] + their own remaining work / throughput.
    
    # A better model: Consider that after starting op i, the critical path
    # through the remaining operations determines the makespan.
    # Estimate: makespan_i = time + min_durations[i] + max(remaining_work[j] for j != i) / num_machines
    # But we don't know num_machines. Use a heuristic.
    
    # Simpler approach: For each candidate i, compute:
    # projected_makespan_i = min_durations[i] + max(remaining_work[j] / remaining_ops[j] for j != i)
    # This approximates: time to finish op i + time to finish the most complex remaining job.
    
    projected_makespans = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # Time to complete op i
        t_i = min_durations[i]
        
        # Estimate remaining completion time for other jobs
        if n > 1:
            others_work = np.delete(remaining_work, i)
            others_ops = np.delete(remaining_ops, i)
            others_avg = others_work / np.maximum(others_ops, 1.0)
            max_remaining = np.max(others_avg) if len(others_avg) > 0 else 0.0
        else:
            max_remaining = 0.0
        
        projected_makespans[i] = t_i + max_remaining

    # We want to MINIMIZE the projected makespan
    # But also consider urgency: jobs with more remaining work should be prioritized
    
    # Combine: minimize projected makespan, but break ties by urgency
    # Use a weighted score where lower is better
    pm_norm = _normalize(projected_makespans)
    
    # Score: minimize projected makespan, maximize urgency
    w_pm = 0.60       # weight on projected makespan (minimize)
    w_work = 0.25      # weight on remaining work (maximize urgency)
    w_spt = 0.15       # weight on SPT (minimize duration)
    
    # Lower score is better
    score = (w_pm * pm_norm + 
             w_work * (1.0 - nw) + 
             w_spt * nd_min)

    # Pick the operation with the lowest score
    best_idx = int(np.argmin(score))
    return best_idx
# EVOLVE-BLOCK-END
