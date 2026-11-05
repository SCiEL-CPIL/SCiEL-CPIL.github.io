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

def _soft_min(values, beta=1.0):
    """Smooth minimum via log-sum-exp (epigraph-style surrogate)."""
    if len(values) == 0:
        return float('inf')
    v = np.array(values, dtype=np.float64)
    v_min = np.min(v)
    # Soft min: v_min - (1/beta) * log(sum(exp(-beta*(v - v_min))))
    # As beta -> inf, this approaches hard min
    scaled = -beta * (v - v_min)
    # Numerical stability
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        max_scaled = 700
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    if sum_exp == 0:
        return v_min
    log_sum = max_scaled + np.log(sum_exp)
    return v_min + log_sum / beta

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

    # Compute per-operation machine completion times and bottleneck capacity
    op_completion_times = []  # list of arrays, one per op
    min_durations = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if mo:
            durations = np.array([d for d in mo.values()])
            min_durations[i] = np.min(durations)
            completions = []
            for m, d in mo.items():
                if m < _MACHINE_COUNT:
                    start = max(time, _machine_load[m])
                    completion = start + d
                else:
                    start = time
                    completion = time + d
                completions.append(completion)
            op_completion_times.append(np.array(completions, dtype=np.float64))
        else:
            min_durations[i] = float('inf')
            op_completion_times.append(np.array([time + op["remaining_work"]], dtype=np.float64))

    # Smooth surrogate scores:
    # 1. Bottleneck capacity: soft-min of completion times (lower is better)
    #    Using soft-min with moderate beta to smooth the hard min
    soft_min_completions = np.zeros(n)
    for i in range(n):
        if len(op_completion_times[i]) > 0:
            soft_min_completions[i] = _soft_min(op_completion_times[i], beta=2.0)
        else:
            soft_min_completions[i] = float('inf')

    # 2. Downstream unlock value: remaining_work normalized
    #    Higher remaining_work means more downstream work unlocked
    max_rw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    unlock_scores = remaining_works / max_rw

    # 3. Flexibility constraint score (higher = more constrained = higher priority)
    max_flex = np.max(flex_scores) if np.max(flex_scores) > 0 else 1.0
    norm_flex = flex_scores / max_flex

    # 4. Duration urgency: shorter operations can be scheduled sooner
    max_dur = np.max(min_durations) if np.max(min_durations) > 0 else 1.0
    dur_urgency = 1.0 - (min_durations / max_dur)  # shorter = higher

    # Composite smooth score:
    # Weighted sum with emphasis on bottleneck completion and downstream unlock
    # The bottleneck capacity concept: an operation's score is inversely related
    # to the soft-min completion time, scaled by how much work it unlocks
    w_completion = 1.0    # weight for soft-min completion (lower is better)
    w_unlock = 0.8        # weight for downstream work unlocked
    w_flex = 0.6          # weight for flexibility constraint
    w_duration = 0.3      # weight for short duration

    # Normalize soft-min completions to [0,1] range (lower completion = higher score)
    min_comp = np.min(soft_min_completions)
    max_comp = np.max(soft_min_completions)
    if max_comp - min_comp > 1e-9:
        norm_completion = 1.0 - (soft_min_completions - min_comp) / (max_comp - min_comp)
    else:
        norm_completion = np.ones(n) * 0.5

    # Composite score (higher is better)
    scores = (
        w_completion * norm_completion +
        w_unlock * unlock_scores +
        w_flex * norm_flex +
        w_duration * dur_urgency
    )

    # Select the operation with the highest composite score
    best_idx = int(np.argmax(scores))

    # Select machine for the chosen operation using smooth congestion-aware selection
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx

    best_machine = None
    best_adjusted_completion = float('inf')
    
    # Smooth congestion penalty: use soft-max of load to encourage load balancing
    # Congestion level per machine: how far past current time the machine is busy
    congestion_levels = np.zeros(_MACHINE_COUNT)
    for m in range(_MACHINE_COUNT):
        congestion_levels[m] = max(0.0, _machine_load[m] - time)
    
    # Soft-max of congestion to get a smooth "system congestion" indicator
    if np.max(congestion_levels) > 0:
        max_cong = np.max(congestion_levels)
        scaled_cong = congestion_levels / max_cong
        # Soft-max weight for each machine's congestion
        cong_weights = np.exp(2.0 * scaled_cong)
        cong_weights /= np.sum(cong_weights)
    else:
        cong_weights = np.ones(_MACHINE_COUNT) / _MACHINE_COUNT

    congestion_alpha = 0.15
    
    for m, d in machine_options.items():
        if m < _MACHINE_COUNT:
            start = max(time, _machine_load[m])
            completion = start + d
            # Smooth congestion penalty weighted by soft-max congestion distribution
            congestion_penalty = congestion_alpha * cong_weights[m] * max(0, _machine_load[m] - time)
            adjusted_completion = completion + congestion_penalty
        else:
            start = time
            completion = time + d
            adjusted_completion = completion

        if adjusted_completion < best_adjusted_completion:
            best_adjusted_completion = adjusted_completion
            best_machine = m

    # Update machine load
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
