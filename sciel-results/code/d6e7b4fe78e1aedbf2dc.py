"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def dispatch(ready_operations, time):
    """
    Dispatch rule that balances:
    1. Urgency: jobs with more remaining work (critical path)
    2. Machine load balancing: prefer machines with less total assigned work
    3. Duration efficiency: prefer shorter processing times when possible
    
    Returns index of the ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Compute scores for each ready operation
    # We want to minimize makespan, so we want to:
    # - Prioritize jobs with more remaining work (urgency)
    # - Prefer machines that are less loaded (balance)
    # - Prefer shorter durations (efficiency)
    
    # Estimate machine loads from remaining_work of all ready ops
    # and the durations they would take on each machine
    
    # For each ready op, find the best machine (shortest duration)
    # and compute a combined score
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        # Remaining work for this job (urgency factor)
        remaining_work = op.get("remaining_work", 0)
        
        # Machine options: {machine: duration}
        machine_options = op.get("machine_options", {})
        
        if not machine_options:
            # No machine options, give low priority
            scores[i] = -1.0
            continue
        
        # Find the shortest duration and average duration
        durations = list(machine_options.values())
        min_duration = min(durations)
        avg_duration = sum(durations) / len(durations)
        
        # Balance factor: prefer machines with lower estimated load
        # We can't directly know machine loads, but we can use
        # the fact that shorter durations generally mean less load impact
        
        # Combined score:
        # - Higher remaining_work -> higher priority (urgency)
        # - Lower min_duration -> higher priority (efficiency)
        # - Lower avg_duration -> higher priority (consistency)
        
        # Normalize to make comparison meaningful
        # Use a weighted combination:
        # score = alpha * remaining_work - beta * min_duration - gamma * avg_duration
        
        alpha = 1.0
        beta = 0.1
        gamma = 0.05
        
        scores[i] = alpha * remaining_work - beta * min_duration - gamma * avg_duration
    
    # Pick the operation with the highest score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
