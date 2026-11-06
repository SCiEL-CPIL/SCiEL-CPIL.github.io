"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 115.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy: Critical Path / WSJF Hybrid
    1. Calculate job urgency as the sum of minimum durations of all remaining operations.
       This approximates the critical path length of the remaining job.
    2. Estimate machine cost as duration adjusted for contention (number of other ready ops 
       that can use that machine).
    3. Select the operation-machine pair that maximizes Urgency / Cost.
       This prioritizes jobs that are "long" (critical) and can be finished quickly on a 
       less contended machine.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    n = len(ready_operations)
    
    # Precompute machine contention: count how many ready ops can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1
    
    # Precompute per-operation features
    # Urgency: Sum of min durations of all remaining ops in the job
    # We approximate "remaining critical path" by summing min durations of remaining ops.
    # This is a lower bound on the time needed to finish the job.
    urgency_scores = np.empty(n, dtype=float)
    
    for i, op in enumerate(ready_operations):
        # remaining_work is typically the sum of durations. 
        # However, we want the *minimum* possible time.
        # If 'remaining_work' is sum of min durations, use it. 
        # Otherwise, we don't have the breakdown. 
        # The prompt says 'remaining_work' is a feature. Let's assume it's a good proxy for load.
        # To be more precise, if we had the list of remaining ops, we'd sum their min durations.
        # Since we only have 'remaining_work' and 'remaining_ops', we use remaining_work.
        # If remaining_work is 0, we use 1.0 to avoid division by zero.
        
        rw = float(op.get("remaining_work", 0.0))
        # A job with more remaining work is more critical.
        urgency_scores[i] = rw if rw > 0 else 1.0

    best_score = -float('inf')
    best_index = 0
    best_machine = None
    
    # Weight for contention penalty
    # Higher beta means we penalize contended machines more heavily
    BETA = 0.5
    
    for i, op in enumerate(ready_operations):
        urgency = urgency_scores[i]
        if urgency <= 0:
            urgency = 1e-6
            
        opts = op["machine_options"]
        if not opts:
            continue

        # Find the best machine for this operation based on the score
        # We evaluate each machine
        # Score = Urgency / Estimated_Cost
        
        # To optimize, we can just find the machine that minimizes Estimated_Cost
        # and then calculate the final score for that pair, 
        # BUT, it's possible that a slightly more expensive machine for a high urgency job
        # yields a better global score than the cheapest machine if the contention is lower?
        # No, for a fixed operation, Urgency is constant. 
        # So we just need the machine with the MINIMUM Estimated_Cost for that operation.
        # Estimated_Cost = d * (1 + BETA * (count - 1))
        
        min_est_cost = float('inf')
        best_m_for_op = None
        
        for m, d in opts.items():
            d = float(d)
            if d <= 0:
                continue
            count = machine_count.get(m, 1)
            est_cost = d * (1.0 + BETA * (count - 1))
            
            if est_cost < min_est_cost:
                min_est_cost = est_cost
                best_m_for_op = m
        
        if best_m_for_op is None:
            continue
            
        # Calculate the final score for this operation
        # We use the best machine's cost for the score comparison
        score = urgency / (min_est_cost + 1e-9)
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Fallback if no valid score was found
    if best_machine is None:
        best_index = 0
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            best_machine = None
    
    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
