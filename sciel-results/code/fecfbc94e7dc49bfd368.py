"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

# Global deadline guard
_START_TIME = _time.time()
_DEADLINE = 110.0  # safety margin before 120s limit


def _soft_min(values, temperature=1.0):
    """Compute soft-min of a list of values using log-sum-exp trick."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0])
    # soft-min = -soft-max of negated values
    neg = -arr
    max_neg = np.max(neg)
    if max_neg > 1e10:
        return float(np.min(arr))
    shifted = neg - max_neg
    sum_exp = np.sum(np.exp(shifted / temperature))
    log_sum = np.log(sum_exp)
    soft_max = max_neg + temperature * log_sum
    return -soft_max


def _soft_max(values, temperature=1.0):
    """Compute soft-max of a list of values using log-sum-exp trick."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0])
    max_val = np.max(arr)
    if max_val > 1e10:
        return float(np.max(arr))
    shifted = arr - max_val
    sum_exp = np.sum(np.exp(shifted / temperature))
    log_sum = np.log(sum_exp)
    return max_val + temperature * log_sum


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Strategy: For each ready operation, estimate the completion time if started
    now on its best machine (considering both duration and machine congestion).
    Use a weighted combination of:
      1. Remaining work of the job (prioritize jobs with more work left)
      2. Estimated completion time on the best machine (prioritize operations
         that can finish sooner)
      3. Machine load balance (avoid overloading already-busy machines)
    
    Returns the index of the ready operation to start next.
    """
    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        return 0
    
    # Check deadline
    if _time.time() - _START_TIME > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    
    # Collect machine options and durations
    machine_loads = {}  # machine -> estimated load (sum of durations of ready ops that could use it)
    for op in ready_operations:
        for m in op["machine_options"]:
            if m not in machine_loads:
                machine_loads[m] = 0.0
            machine_loads[m] += op["machine_options"][m]
    
    # For each ready operation, compute a composite score
    scores = []
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Estimate completion time for each machine option
        # We use the duration plus a congestion penalty based on how many
        # other ready operations also target this machine
        machine_scores = []
        for m, dur in machine_options.items():
            # Congestion: how many other ops target this machine
            congestion_count = 0
            for j, other_op in enumerate(ready_operations):
                if j != i and m in other_op["machine_options"]:
                    congestion_count += 1
            
            # Weighted duration: base duration + congestion penalty
            # Use soft-min across machines for a smooth estimate
            congestion_penalty = congestion_count * dur * 0.1
            est_completion = dur + congestion_penalty
            machine_scores.append(est_completion)
        
        # Soft-min completion time across machines (smooth min)
        if machine_scores:
            soft_min_completion = _soft_min(machine_scores, temperature=0.5)
            # Also get the best (min) machine for tie-breaking
            best_machine_idx = int(np.argmin(machine_scores))
        else:
            soft_min_completion = remaining_work
            best_machine_idx = 0
        
        # Composite score: we want to minimize this
        # Component 1: remaining work (higher = more urgent)
        # Component 2: estimated completion time (lower = better)
        # Component 3: number of remaining ops (more ops = more complex job)
        
        # Normalize components
        rem_work_norm = remaining_work / max(1.0, remaining_work)
        completion_norm = soft_min_completion / max(1.0, soft_min_completion)
        rem_ops_norm = remaining_ops / max(1, remaining_ops)
        
        # We want to pick the operation that:
        # - Has high remaining work (urgent)
        # - Can complete soon (low completion time)
        # - Is part of a job with many remaining ops (to keep jobs flowing)
        
        # Score: lower is better
        # Weighted combination favoring operations that reduce makespan
        score = (
            0.4 * (1.0 - rem_work_norm) +   # prefer high remaining work
            0.4 * completion_norm +           # prefer low completion time
            0.2 * rem_ops_norm                # prefer jobs with more remaining ops
        )
        
        scores.append(score)
    
    # Pick the operation with the lowest score
    best_idx = int(np.argmin(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
