"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

# Global state for load balancing
# Assuming 5 machines based on task description
_MACHINE_COUNT = 5
_machine_load = np.zeros(_MACHINE_COUNT)  # busy-until time per machine
_last_seen_time = -1.0

def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Args:
        ready_operations: List of dicts with job, op, machine_options, remaining_ops, remaining_work
        time: Current simulation time
        
    Returns:
        Index of the operation to start, or (index, machine) tuple.
    """
    global _machine_load, _last_seen_time

    if not ready_operations:
        return 0

    # Deadline check
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: Simple Most Remaining Work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Detect new instance: if time resets, reset load counters
    if time < _last_seen_time:
        _machine_load = np.zeros(_MACHINE_COUNT)
    _last_seen_time = time

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick the machine that gives earliest completion
            best_machine = None
            best_completion = float('inf')
            for m, d in machine_options.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                if completion < best_completion:
                    best_completion = completion
                    best_machine = m
            if best_machine is not None and best_machine < _MACHINE_COUNT:
                _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations])
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations])
    
    # Flexibility: 1 / number of machine options.
    # Less options -> Higher priority (constrained)
    flex_scores = np.array([
        1.0 / len(op["machine_options"]) if op["machine_options"] else 0.0
        for op in ready_operations
    ])

    # Compute earliest completion time for each operation
    earliest_completions = np.zeros(n)
    min_durations = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            min_durations[i] = min(mo.values())
            best_comp = float('inf')
            for m, d in mo.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                if completion < best_comp:
                    best_comp = completion
            earliest_completions[i] = best_comp
        else:
            min_durations[i] = float('inf')
            earliest_completions[i] = time + op["remaining_work"]

    # Smooth bottleneck-aware scoring: combine flexibility, urgency, and machine contention
    # using log-sum-exp for a smooth surrogate of the min-completion objective
    
    # Compute per-operation minimum completion across eligible machines
    # and a soft-min that captures contention effects
    beta_flex = 2.0    # weight for flexibility (constraint-first)
    beta_work = 1.5    # weight for remaining work urgency
    beta_comp = 3.0    # weight for completion time (bottleneck awareness)
    
    # Normalize features to [0,1] range for stable scoring
    min_work = np.min(remaining_works)
    max_work = np.max(remaining_works)
    work_range = max_work - min_work + 1e-9
    norm_works = (remaining_works - min_work) / work_range
    
    min_comp = np.min(earliest_completions)
    max_comp = np.max(earliest_completions)
    comp_range = max_comp - min_comp + 1e-9
    norm_comps = (earliest_completions - min_comp) / comp_range
    
    # Soft-min of completion times: lower is better, captures contention
    # Use log-sum-exp for numerical stability
    comp_shifted = -earliest_completions  # negate so max becomes min
    comp_max = np.max(comp_shifted)
    exp_shifted = np.exp(comp_shifted - comp_max)
    log_sum_exp = comp_max + np.log(np.sum(exp_shifted))
    soft_min_comp = -log_sum_exp  # back to original scale
    
    # Score: higher is better
    # Flexibility: more constrained ops get higher score
    flex_norm = flex_scores / (np.max(flex_scores) + 1e-9)
    
    # Urgency: more remaining work gets higher score
    urgency_score = norm_works
    
    # Bottleneck: lower soft-min completion gets higher score
    bottleneck_score = 1.0 - (soft_min_comp - min_comp) / comp_range
    bottleneck_score = np.clip(bottleneck_score, 0, 1)
    
    # Combined score with smooth weighting
    scores = (beta_flex * flex_norm + 
              beta_work * urgency_score + 
              beta_comp * bottleneck_score)
    
    best_idx = int(np.argmax(scores))

    # Select machine for the chosen operation: use look-ahead bottleneck heuristic
    # Score each eligible machine by its projected load after assignment,
    # preferring machines that minimize the resulting max-load imbalance
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx

    best_machine = None
    best_score = float('inf')
    
    # Look-ahead: estimate impact on system bottleneck
    # For each candidate machine, compute what the max machine load would be
    # if we assign this operation there
    current_max_load = np.max(_machine_load) if _MACHINE_COUNT > 0 else 0
    
    for m, d in machine_options.items():
        if m < _MACHINE_COUNT:
            start = max(time, _machine_load[m])
            new_load = start + d
            
            # Projected max load if assigned to machine m
            projected_max = max(current_max_load, new_load)
            
            # Penalty: how much does this assignment increase the bottleneck?
            bottleneck_increase = max(0, projected_max - current_max_load)
            
            # Also consider immediate completion time
            completion = start + d
            idle_time = max(0, start - time)  # how long op waits
            
            # Combined machine score: minimize bottleneck increase + idle + completion
            machine_score = (1.0 * bottleneck_increase + 
                           0.3 * idle_time + 
                           0.5 * completion)
        else:
            start = time
            completion = time + d
            machine_score = completion

        if machine_score < best_score:
            best_score = machine_score
            best_machine = m

    # Update machine load
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
