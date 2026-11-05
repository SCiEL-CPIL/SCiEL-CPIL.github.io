"""Job-shop dispatch rule: LRPT-dominant with smooth machine-contention tiebreak."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0  # safety margin well before the 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Dominant term: Remaining Work (LRPT) to clear critical paths.
    2. Smooth contention penalty: penalize operations targeting heavily loaded machines.
    3. Urgency bonus: slight preference for jobs with fewer remaining operations.
    4. Duration penalty: slight preference for shorter operations to reduce queue length.
    
    The contention term uses a soft-min over machine options to create a smooth surrogate
    for the nonsmooth min operation, improving gradient-like behavior in the heuristic.
    """
    if not ready_operations:
        return 0
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # Step 1: Count how many ready operations target each machine (contention proxy)
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    max_count = max(machine_counts.values()) if machine_counts else 1
    if max_count == 0:
        max_count = 1

    # Weights
    W_RW = 1.0        # Weight for Remaining Work (LRPT) - dominant
    W_URGENCY = 0.5   # Weight for Urgency (1/remaining_ops)
    W_CONTENTION = 0.2  # Weight for machine contention penalty
    W_DURATION = 0.05   # Weight for duration penalty

    best_idx = 0
    best_score = float("-inf")

    for i in range(n):
        op = ready_operations[i]
        rw = float(op.get("remaining_work", 0) or 0)
        rem_ops = int(op.get("remaining_ops", 1) or 1)
        
        if rem_ops < 1:
            rem_ops = 1

        mo = op.get("machine_options", {})
        if not mo:
            score = W_RW * rw + W_URGENCY / rem_ops
        else:
            # Compute smooth contention metric using soft-min over machine options
            # For each machine option, compute a "cost" that combines contention and duration
            # Use log-sum-exp for soft-min to create a smooth surrogate
            
            costs = []
            for m, d in mo.items():
                c = machine_counts.get(m, 1)
                # Cost combines contention (higher is worse) and duration
                # Normalize contention to [0, 1]
                norm_c = c / max_count
                # Weight duration by a small factor
                cost = norm_c + 0.1 * (d / (rw + 1.0)) if rw > 0 else norm_c
                costs.append(cost)
            
            # Soft-min using log-sum-exp with negative temperature
            # soft_min(x) = -log(sum(exp(-x_i / T))) * T
            T = 0.1  # Temperature for soft-min
            if len(costs) == 1:
                soft_min_cost = costs[0]
            else:
                neg_costs = [-c / T for c in costs]
                max_neg = max(neg_costs)
                sum_exp = sum(math.exp(x - max_neg) for x in neg_costs)
                log_sum_exp = max_neg + math.log(sum_exp) if sum_exp > 0 else max_neg
                soft_min_cost = -log_sum_exp * T
            
            # Urgency term: smaller remaining ops count is better (higher score)
            urgency = W_URGENCY / rem_ops
            
            # Duration penalty: use the minimum duration as a proxy
            min_dur = min(mo.values())
            dur_penalty = W_DURATION * min_dur
            
            # Contention penalty: higher soft_min_cost is worse (lower score)
            contention_penalty = W_CONTENTION * soft_min_cost
            
            score = W_RW * rw + urgency - contention_penalty - dur_penalty

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)

# EVOLVE-BLOCK-END
