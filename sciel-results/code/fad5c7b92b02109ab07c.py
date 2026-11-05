"""Improved dispatch rule for Flexible Job-Shop Scheduling."""

import math

def dispatch(ready_operations, time):
    """
    Picks the ready operation to start next and the best machine for it.
    
    Strategy:
    1. For each ready operation, use a soft-min surrogate to estimate effective processing time,
       which smoothly accounts for all machine options rather than just the best one.
    2. Calculate a priority score combining urgency (remaining work) and efficiency,
       with a machine load balancing term to avoid bottlenecks.
    3. Select the operation with the highest score.
    4. Return the index of the operation and the ID of the best machine.
    """
    if not ready_operations:
        return 0

    # Temperature parameter for soft-min: lower = closer to hard min
    # Higher temperature smooths out the selection, considering all machines
    temp = 2.0
    
    best_index = 0
    best_score = -float('inf')
    best_machine = None
    best_min_dur = float('inf')

    for i, op in enumerate(ready_operations):
        remaining_work = op.get('remaining_work', 0)
        machine_options = op.get('machine_options', {})
        
        if not machine_options:
            continue

        # Soft-min: -temp * log(sum(exp(-duration/temp)))
        # This gives a smooth approximation to min(duration) that considers all options
        log_sum = 0.0
        min_duration = float('inf')
        min_machine = None
        for machine_id, duration in machine_options.items():
            if duration < min_duration:
                min_duration = duration
                min_machine = machine_id
            # Use log-sum-exp for numerical stability
            log_sum += math.exp(-duration / temp)
        
        if min_machine is None:
            continue
        
        # Soft-min effective duration
        soft_min_dur = -temp * math.log(log_sum)
        
        # Machine load balancing: penalize if the best machine is likely busy
        # Use remaining_ops as a proxy for how many more operations will compete
        remaining_ops = op.get('remaining_ops', 1)
        load_factor = 1.0 + 0.1 * remaining_ops  # Slight penalty for jobs with many remaining ops
        
        # Score: High remaining work, low effective duration, balanced load
        epsilon = 1e-6
        urgency = remaining_work / (soft_min_dur + epsilon)
        efficiency = 1.0 / (soft_min_dur + epsilon)
        
        # Combine: urgency is primary, efficiency as secondary
        score = urgency * 0.7 + efficiency * 0.3
        score /= load_factor
        
        # Tie-breaking: Prefer shorter effective duration, then fewer remaining ops
        if score > best_score + 1e-12:
            best_score = score
            best_index = i
            best_machine = min_machine
            best_min_dur = soft_min_dur
        elif abs(score - best_score) <= 1e-12:
            # Prefer shorter soft-min duration
            if soft_min_dur < best_min_dur - 1e-9:
                best_index = i
                best_machine = min_machine
                best_min_dur = soft_min_dur
            elif abs(soft_min_dur - best_min_dur) <= 1e-9:
                # Prefer fewer remaining operations (less future contention)
                if remaining_ops < op.get('remaining_ops', float('inf')):
                    best_index = i
                    best_machine = min_machine
                    best_min_dur = soft_min_dur

    return (best_index, best_machine)
