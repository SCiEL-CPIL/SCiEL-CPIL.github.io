"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

# Global state for load balancing
_MACHINE_COUNT = 5
_machine_load = np.zeros(_MACHINE_COUNT)
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

    # Look-ahead bottleneck heuristic:
    # For each operation, compute the minimum completion time if assigned to the
    # least-loaded eligible machine (among its options). This captures immediate
    # machine contention impact rather than just static job attributes.
    
    # Precompute machine availability (when each machine is free)
    machine_available = np.array([max(time, _machine_load[m]) for m in range(_MACHINE_COUNT)])
    
    # For each ready operation, compute:
    # 1. Look-ahead completion: min over eligible machines of (machine_available[m] + duration)
    #    This simulates assigning to the best (least-loaded) eligible machine
    # 2. Remaining work (urgency)
    # 3. Flexibility (constraint level)
    
    look_ahead_completions = np.zeros(n)
    remaining_works = np.zeros(n)
    flex_scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        remaining_works[i] = op["remaining_work"]
        
        if mo:
            flex_scores[i] = 1.0 / len(mo)
            # Look-ahead: assign to the machine that gives earliest completion
            min_comp = float('inf')
            for m, d in mo.items():
                if m < _MACHINE_COUNT:
                    comp = max(time, _machine_load[m]) + d
                else:
                    comp = time + d
                if comp < min_comp:
                    min_comp = comp
            look_ahead_completions[i] = min_comp
        else:
            look_ahead_completions[i] = time + op["remaining_work"]
    
    # Composite score: we want to minimize makespan.
    # The look-ahead completion time directly estimates when this operation finishes
    # if we pick it now and assign it to its best available machine.
    # Operations that finish sooner reduce the critical path.
    #
    # Strategy: prioritize operations with the EARLIEST look-ahead completion
    # (they're on the critical path and finishing them unlocks successors quickly),
    # broken by flexibility (more constrained ops get priority to avoid deadlock),
    # then by remaining work.
    
    # Use a weighted score: primarily look-ahead completion, secondarily flexibility
    # We want to minimize completion, so lower is better.
    # Flexibility: higher flex_score means more constrained (fewer options), so we
    # should prioritize those. Subtract a small multiple of flex_score to boost constrained ops.
    
    # Normalize for stability
    max_comp = np.max(look_ahead_completions) if n > 0 else 1.0
    min_comp = np.min(look_ahead_completions) if n > 0 else 0.0
    comp_range = max_comp - min_comp if max_comp > min_comp else 1.0
    
    # Primary: look-ahead completion time (lower = better)
    # Secondary: flexibility (higher = more constrained = higher priority)
    # Tertiary: remaining work (higher = more urgent)
    
    # Lexicographic: first by look-ahead completion, then by flexibility, then by remaining work
    # But to avoid pure lexicographic brittleness, use a smooth composite:
    
    # Score = look_ahead_completion - alpha * flex_score * comp_range - beta * remaining_work
    # Lower score = higher priority
    
    alpha = 0.1  # Flexibility weight
    beta = 0.05  # Remaining work weight
    
    scores = look_ahead_completions - alpha * flex_scores * comp_range - beta * remaining_work
    
    best_idx = int(np.argmin(scores))
    
    # Select machine for the chosen operation: minimize completion time
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx
    
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
    
    # Update machine load
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
