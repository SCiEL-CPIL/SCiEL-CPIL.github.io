"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

# Track machine loads across dispatch calls for look-ahead contention estimation
_machine_loads = {}


def _soft_min(values, temperature=1.0):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    v = np.array(values, dtype=float)
    v_min = np.min(v)
    # Soft min: -temperature * log(sum(exp(-(v - v_min)/temperature))) + v_min
    exp_vals = np.exp(-(v - v_min) / temperature)
    return v_min - temperature * np.log(np.sum(exp_vals))


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: pick operation with most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick least-loaded eligible machine
            best_machine = min(machine_options, key=lambda m: _machine_loads.get(m, 0.0))
            return (0, best_machine)
        return 0

    # Extract key features for all ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ])
    
    # Compute critical path-like estimate
    avg_work_per_op = np.where(remaining_ops > 0, remaining_works / np.maximum(remaining_ops, 1), remaining_works)
    
    # LOOK-AHEAD BOTTLENECK HEURISTIC:
    # For each operation, estimate its effective completion time if assigned
    # to the currently least-loaded eligible machine.
    # This captures immediate machine contention impact.
    
    # Build machine load estimates from current time context
    # We use the current simulation time as a proxy for when machines become free
    # and track relative loads via _machine_loads dict
    
    # For each operation, compute the look-ahead completion time
    # on its best (least-loaded) eligible machine
    look_ahead_completion_times = np.zeros(n)
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            look_ahead_completion_times[i] = np.inf
            best_machines[i] = None
            continue
        
        # For each eligible machine, estimate completion time:
        # current machine load (from _machine_loads) + processing time on that machine
        # The machine with minimum estimated completion time is the best choice
        best_completion_time = np.inf
        best_machine = None
        
        for machine, duration in machine_options.items():
            machine_load = _machine_loads.get(machine, 0.0)
            # Estimated completion time = machine's current load + this op's duration
            estimated_completion = machine_load + duration
            if estimated_completion < best_completion_time:
                best_completion_time = estimated_completion
                best_machine = machine
        
        look_ahead_completion_times[i] = best_completion_time
        best_machines[i] = best_machine
    
    # Also compute a soft-min version to capture the spread of machine options
    # (if an op has multiple good machines, it's less bottlenecked)
    soft_min_completion_times = np.zeros(n)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            soft_min_completion_times[i] = np.inf
            continue
        completion_options = []
        for machine, duration in machine_options.items():
            machine_load = _machine_loads.get(machine, 0.0)
            completion_options.append(machine_load + duration)
        soft_min_completion_times[i] = _soft_min(completion_options, temperature=0.5)
    
    # Normalize for scoring
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_avg = np.max(avg_work_per_op) if np.max(avg_work_per_op) > 0 else 1.0
    max_inv_dur = np.max(1.0 / min_durations) if np.any(min_durations > 0) else 1.0
    
    # Look-ahead completion time: lower is better, so invert for scoring
    # Normalize: 1 / (1 + completion_time) gives higher score for faster completion
    normalized_look_ahead = 1.0 / (1.0 + look_ahead_completion_times / (max_work + 1e-9))
    
    # Soft-min version: captures flexibility (lower soft-min = more options available)
    normalized_soft_min = 1.0 / (1.0 + soft_min_completion_times / (max_work + 1e-9))
    
    # Normalize other components
    normalized_urgency = remaining_works / (max_work + 1e-9)
    normalized_avg = avg_work_per_op / (max_avg + 1e-9)
    normalized_efficiency = (1.0 / min_durations) / (max_inv_dur + 1e-9)
    
    max_remaining_ops = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    normalized_complete = 1.0 - (remaining_ops / (max_remaining_ops + 1e-9))
    
    # Weighted score combining machine-centric look-ahead with job-centric urgency:
    # - Look-ahead bottleneck: primary signal (machine contention impact)
    # - Soft-min flexibility: secondary (captures machine optionality)
    # - Urgency: total remaining work
    # - Depth-adjusted urgency: average work per remaining op
    # - Efficiency: how fast on best machine (static)
    # - Completeness: closer to completion
    w_look_ahead = 1.2
    w_soft_min = 0.3
    w_urgency = 0.8
    w_avg = 0.3
    w_efficiency = 0.2
    w_complete = 0.15
    
    scores = (w_look_ahead * normalized_look_ahead + w_soft_min * normalized_soft_min +
              w_urgency * normalized_urgency + w_avg * normalized_avg +
              w_efficiency * normalized_efficiency + w_complete * normalized_complete)

    best_idx = int(np.argmax(scores))

    # Choose the best machine for the selected operation (least-loaded)
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        best_machine = min(machine_options, key=lambda m: _machine_loads.get(m, 0.0))
        
        # Update machine load tracking for future look-ahead
        duration = machine_options[best_machine]
        _machine_loads[best_machine] = _machine_loads.get(best_machine, 0.0) + duration
        
        # Decay other machines' loads slightly (simulating time passing)
        for m in _machine_loads:
            if m != best_machine:
                _machine_loads[m] = max(0.0, _machine_loads[m] - 0.1)
        
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
