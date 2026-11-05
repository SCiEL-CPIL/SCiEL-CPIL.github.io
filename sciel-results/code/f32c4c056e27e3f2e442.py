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

    # For each ready operation, compute:
    # 1. Earliest completion time if assigned to best available machine
    # 2. Remaining work (urgency)
    # 3. Flexibility (constraint level)

    earliest_completions = np.zeros(n)
    min_durations = np.zeros(n)
    flex_scores = np.zeros(n)
    remaining_works = np.zeros(n)

    for i, op in enumerate(ready_operations):
        remaining_works[i] = op["remaining_work"]
        mo = op["machine_options"]
        if mo:
            num_opts = len(mo)
            flex_scores[i] = 1.0 / num_opts
            min_durations[i] = min(mo.values())
            # Find machine giving earliest completion
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
            flex_scores[i] = 0.0
            min_durations[i] = float('inf')
            earliest_completions[i] = time + op["remaining_work"]

    # Normalize features
    max_work = np.max(remaining_works)
    max_flex = np.max(flex_scores)
    max_comp = np.max(earliest_completions)
    min_comp = np.min(earliest_completions)

    if max_work > 0:
        norm_work = remaining_works / max_work
    else:
        norm_work = remaining_works

    if max_flex > 0:
        norm_flex = flex_scores / max_flex
    else:
        norm_flex = flex_scores

    # Normalize completion time: lower is better, so invert
    comp_range = max_comp - min_comp
    if comp_range > 0:
        norm_comp = 1.0 - (earliest_completions - min_comp) / comp_range  # higher = earlier = better
    else:
        norm_comp = np.ones(n)

    # Weighted score: urgency + constraint + early completion potential
    w_work = 0.8
    w_flex = 0.3
    w_comp = 1.2
    scores = w_work * norm_work + w_flex * norm_flex + w_comp * norm_comp

    # Select best operation
    best_idx = int(np.argmax(scores))

    # Tie-breaking: prefer earlier completion, then shorter duration
    max_score = scores[best_idx]
    close_indices = [i for i, s in enumerate(scores) if abs(s - max_score) < 1e-5]

    if len(close_indices) > 1:
        # Sort by earliest completion, then min duration
        close_indices.sort(key=lambda i: (earliest_completions[i], min_durations[i]))
        best_idx = close_indices[0]

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
