"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

# Global state for load balancing
_MACHINE_COUNT = 5
_machine_load = np.zeros(_MACHINE_COUNT)  # busy-until time per machine
_last_seen_time = -1.0

# Flow-network inspired state:
# Track per-job remaining work to estimate "distance to sink"
_job_remaining_work = {}  # job_id -> remaining work
_job_total_work = {}      # job_id -> total work (first seen)
_machine_capacity_used = np.zeros(_MACHINE_COUNT)  # cumulative processing assigned

def _reset_state():
    global _machine_load, _job_remaining_work, _job_total_work, _machine_capacity_used
    _machine_load = np.zeros(_MACHINE_COUNT)
    _job_remaining_work = {}
    _job_total_work = {}
    _machine_capacity_used = np.zeros(_MACHINE_COUNT)

def _soft_min(values, beta=5.0):
    """
    Smooth minimum via log-sum-exp (epigraph-style surrogate).
    As beta -> inf, approaches hard min. Smaller beta gives smoother result.
    """
    if len(values) == 0:
        return float('inf')
    v = np.array(values, dtype=np.float64)
    v_min = np.min(v)
    # Soft min: v_min - (1/beta) * log(sum(exp(-beta*(v - v_min))))
    # This is approximately v_min for large beta, but smooth
    scaled = -beta * (v - v_min)
    # Numerical stability: subtract max
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        max_scaled = 700
    sum_exp = np.sum(np.exp(scaled - max_scaled))
    if sum_exp == 0:
        return v_min
    log_sum = max_scaled + np.log(sum_exp)
    return v_min + log_sum / beta

def _soft_max(values, beta=5.0):
    """
    Smooth maximum via log-sum-exp.
    """
    if len(values) == 0:
        return float('-inf')
    v = np.array(values, dtype=np.float64)
    v_max = np.max(v)
    scaled = beta * (v - v_max)
    min_scaled = np.min(scaled)
    if min_scaled < -700:
        min_scaled = -700
    sum_exp = np.sum(np.exp(scaled - min_scaled))
    if sum_exp == 0:
        return v_max
    log_sum = min_scaled + np.log(sum_exp)
    return v_max + log_sum / beta

def dispatch(ready_operations, time):
    """
    Dispatch rule using flow-network bottleneck capacity model with smooth surrogates.
    Each operation's priority = smooth bottleneck capacity along the path
    from job source -> operation -> machine sink, weighted by downstream unlock.
    Uses log-sum-exp soft-min for the bottleneck to get a smooth optimization surface.
    """
    global _machine_load, _last_seen_time, _job_remaining_work, _job_total_work, _machine_capacity_used

    if not ready_operations:
        return 0

    # Deadline check
    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Detect new instance
    if time < _last_seen_time:
        _reset_state()
    _last_seen_time = time

    # Update job tracking
    for op in ready_operations:
        j = op["job"]
        if j not in _job_total_work:
            _job_total_work[j] = op["remaining_work"]
        _job_remaining_work[j] = min(_job_remaining_work.get(j, op["remaining_work"]), op["remaining_work"])

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
                _machine_capacity_used[best_machine] += machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Flow-network model with smooth surrogates:
    # For each ready operation, compute:
    # 1. Source capacity: how much work remains in the job (higher = more critical)
    # 2. Edge capacity: earliest completion time on best machine (lower = better)
    # 3. Sink capacity: machine's residual capacity (lower load = better)
    # 4. Bottleneck: soft-min of normalized capacities along the path
    
    scores = np.zeros(n)
    earliest_completions = np.zeros(n)
    best_machines = [None] * n
    source_caps = np.zeros(n)
    sink_caps = np.zeros(n)
    flex_vals = np.zeros(n)
    unlock_vals = np.zeros(n)
    
    # First pass: compute all components
    for i, op in enumerate(ready_operations):
        j = op["job"]
        rw = op["remaining_work"]
        mo = op["machine_options"]
        
        if not mo:
            scores[i] = -float('inf')
            earliest_completions[i] = float('inf')
            continue
        
        # Source capacity: remaining work in job (normalized)
        total_j = _job_total_work.get(j, rw)
        source_cap = rw / max(total_j, 1.0) if total_j > 0 else 0.0
        source_caps[i] = source_cap
        
        # Edge capacity: earliest completion time
        best_comp = float('inf')
        best_m = None
        for m, d in mo.items():
            if m < _MACHINE_COUNT:
                start = max(time, _machine_load[m])
                comp = start + d
            else:
                comp = time + d
            if comp < best_comp:
                best_comp = comp
                best_m = m
        
        earliest_completions[i] = best_comp
        best_machines[i] = best_m
        
        # Sink capacity: machine's current load relative to average
        if best_m is not None and best_m < _MACHINE_COUNT:
            avg_load = np.mean(_machine_load)
            # Normalize: 1.0 if machine is idle or below average, 0.0 if very loaded
            sink_cap = 1.0 - min(1.0, max(0.0, (_machine_load[best_m] - avg_load) / max(avg_load, 1.0)))
        else:
            sink_cap = 0.5
        sink_caps[i] = sink_cap
        
        # Downstream unlock: remaining work after this op (more unlock = more critical)
        min_duration = min(d for d in mo.values())
        remaining_after = max(0, rw - min_duration)
        unlock_val = remaining_after / max(total_j, 1.0)
        unlock_vals[i] = unlock_val
        
        # Flexibility: fewer options = more constrained = higher priority
        flex = 1.0 / len(mo)
        flex_vals[i] = flex

    # Second pass: compute normalized edge_inv using global min/max
    valid_ecs = earliest_completions[np.isfinite(earliest_completions)]
    if len(valid_ecs) > 0:
        ec_min = np.min(valid_ecs)
        ec_max = np.max(valid_ecs)
        ec_range = ec_max - ec_min
    else:
        ec_min = 0.0
        ec_max = 1.0
        ec_range = 1.0
    
    # Compute scores with smooth bottleneck
    for i in range(n):
        if not np.isfinite(earliest_completions[i]):
            scores[i] = -float('inf')
            continue
        
        # Edge inv: lower completion = higher value (normalized to [0,1])
        if ec_range > 1e-9:
            edge_inv = 1.0 - (earliest_completions[i] - ec_min) / ec_range
        else:
            edge_inv = 0.5
        
        # Smooth bottleneck: use soft-min of the three capacity terms
        # This is a smooth surrogate for the hard min, enabling better gradient flow
        cap_values = [source_caps[i], edge_inv, sink_caps[i]]
        bottleneck = _soft_min(cap_values, beta=8.0)
        
        # Weighted combination: emphasize bottleneck, unlock, and flexibility
        # Higher weights on bottleneck since it captures the critical path constraint
        scores[i] = 1.5 * bottleneck + 0.8 * unlock_vals[i] + 0.4 * flex_vals[i]

    # Select best operation
    best_idx = int(np.argmax(scores))
    
    # Machine selection: minimize completion with look-ahead congestion penalty
    machine_options = ready_operations[best_idx]["machine_options"]
    if not machine_options:
        return best_idx
    
    best_machine = None
    best_score = float('inf')
    avg_load = np.mean(_machine_load)
    
    # Look-ahead: estimate how many other ready ops would compete for each machine
    machine_competition = np.zeros(_MACHINE_COUNT)
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            if m < _MACHINE_COUNT:
                machine_competition[m] += 1
    
    for m, d in machine_options.items():
        if m < _MACHINE_COUNT:
            start = max(time, _machine_load[m])
            completion = start + d
            # Look-ahead congestion penalty: penalize machines with high current load
            # AND high competition from other ready operations
            load_penalty = 0.3 * max(0, _machine_load[m] - avg_load)
            competition_penalty = 0.1 * max(0, machine_competition[m] - 1)
            score = completion + load_penalty + competition_penalty
        else:
            score = time + d
        
        if score < best_score:
            best_score = score
            best_machine = m
    
    # Update state
    if best_machine is not None and best_machine < _MACHINE_COUNT:
        _machine_load[best_machine] = max(time, _machine_load[best_machine]) + machine_options[best_machine]
        _machine_capacity_used[best_machine] += machine_options[best_machine]
    
    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
