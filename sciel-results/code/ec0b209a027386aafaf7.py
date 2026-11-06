"""Job-Shop dispatch via non-linear urgency-speed priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _min_pair(machine_options):
    """Return (min_duration, machine) from {machine: duration}."""
    if not machine_options:
        return 1.0, None
    best_m, best_d = None, float("inf")
    for m, d in machine_options.items():
        if d < best_d:
            best_d = d
            best_m = m
    return best_d, best_m


def _is_time_up():
    return (_time.time() - _START_TIME) > _TIME_LIMIT


def dispatch(ready_operations, time):
    """
    Dispatch rule using non-linear urgency-speed priority.
    
    Scores each ready operation based on:
    1. Urgency: remaining_work (proxy for critical path length)
    2. Speed: 1 / min_duration (proxy for how quickly it frees a resource)
    
    The core heuristic is the product of urgency and speed (urgency / duration),
    which favors completing critical jobs on fast machines quickly.
    This structural change replaces simulation-based search with direct evaluation.
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _TIME_LIMIT:
        return 0
    
    n = len(ready_operations)
    
    if n == 1:
        return 0

    best_idx = 0
    best_score = -float("inf")
    
    for i, op in enumerate(ready_operations):
        min_dur, min_mach = _min_pair(op["machine_options"])
        
        # Avoid division by zero
        if min_dur <= 0:
            min_dur = 1e-9
            
        # Urgency: Higher remaining work means more critical.
        urgency = op["remaining_work"]
        
        # Speed: Higher is better (shorter duration).
        # The "urgency-speed product" is urgency * (1/duration) = urgency / duration.
        score = urgency / min_dur
        
        # Refinement: 
        # 1. Prefer ops with fewer remaining ops? 
        #    Jobs closer to completion might be prioritized to clear them.
        #    However, long jobs (high remaining work) are usually more critical for makespan.
        #    Let's add a small bonus for "progress" (lower remaining_ops) to break ties
        #    or prioritize finishing jobs if urgency is similar.
        rem_ops = op["remaining_ops"]
        if rem_ops > 0:
            score += 5.0 / rem_ops
        
        # 2. Tie-breaker: Shortest Processing Time (SPT)
        #    If scores are very close, prefer shorter duration to reduce machine blockage.
        score -= 0.01 * min_dur
        
        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
