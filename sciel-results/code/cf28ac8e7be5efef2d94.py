"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

# Module-level deadline guard
_START_TIME = _time.time()
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy: Dynamic bottleneck optimization.
    Prioritize the operation that maximizes a score balancing:
    1. Job urgency: remaining_work (critical path proxy)
    2. Machine congestion avoidance: prefer machines with less estimated load
    3. Operation efficiency: prefer shorter processing times when possible
    
    Score = remaining_work / (1 + min_duration) * (1 / (1 + machine_load_proxy))
    
    The machine_load_proxy is approximated by the minimum duration across
    eligible machines (shorter = less congestion impact).
    
    Returns index of the selected ready operation.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    
    # If only one option, return it
    if n == 1:
        return 0
    
    # Precompute scores for each ready operation
    # We want to maximize: urgency / congestion_penalty
    # urgency = remaining_work
    # congestion_penalty relates to how "expensive" this op is on its best machine
    
    best_idx = 0
    best_score = -1.0
    
    # Soft-min parameter. Higher beta -> closer to true min.
    # 5.0 is a reasonable balance for typical integer/float durations.
    beta = 5.0
    
    for i, op in enumerate(ready_operations):
        remaining_work = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            continue
        
        # Calculate soft-min of durations
        # soft_min = -1/beta * ln(sum(exp(-beta * d)))
        # This is more robust than hard min as it accounts for the spread of options
        durs = list(machine_options.values())
        
        if len(durs) == 1:
            soft_min_dur = durs[0]
        else:
            # Scale to prevent overflow/underflow in exp
            # Find the min to use as reference for stable log-sum-exp
            min_d = min(durs)
            if min_d <= 0:
                min_d = 1e-9
            
            # exp(-beta * (d - min_d)) is stable
            # sum_term = sum(exp(-beta * (d - min_d)))
            # soft_min = min_d - (1/beta) * ln(sum_term)
            
            sum_term = 0.0
            for d in durs:
                # d - min_d >= 0, so -beta * (d - min_d) <= 0
                # exp of negative number is safe
                sum_term += math.exp(-beta * (d - min_d))
            
            if sum_term == 0:
                soft_min_dur = min_d
            else:
                soft_min_dur = min_d - (1.0 / beta) * math.log(sum_term)
        
        # Avoid division by zero
        if soft_min_dur <= 0:
            soft_min_dur = 1e-9
        
        # The ratio: more remaining work is better, but longer effective duration is worse
        ratio = remaining_work / soft_min_dur
        
        # Additional tiebreaker: prefer operations with more remaining_ops in their job
        remaining_ops = op.get("remaining_ops", 0)
        
        # Combined score: primary is ratio, secondary is remaining_ops
        # Slightly increased weight on ratio compared to ops count
        score = ratio + 0.05 * remaining_ops
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
