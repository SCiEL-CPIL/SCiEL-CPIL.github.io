"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TIME = _time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _ensure_deadline():
    """Return True if we are within the wall-clock budget."""
    return (_time.time() - _START_TIME) < _DEADLINE


def _compute_op_score(op, time, machine_loads):
    """
    Compute a priority score for a ready operation.
    Higher score means higher priority (start first).
    
    Primary: Longest Processing Time (LPT) on the least-loaded candidate machine
    Secondary: Machine congestion avoidance (soft-min of loads)
    Tertiary: Job urgency (fewer remaining ops)
    """
    machine_options = op["machine_options"]
    if not machine_options:
        return 0.0
    
    machines = list(machine_options.keys())
    durations = np.array([machine_options[m] for m in machines], dtype=float)
    num_machines = len(machines)
    
    # Get current loads for candidate machines
    loads = np.array([machine_loads.get(m, 0.0) for m in machines], dtype=float)
    
    # Primary: LPT - pick the operation with the longest duration
    # This directly addresses bottleneck operations
    max_dur = float(np.max(durations))
    
    # Secondary: Machine congestion - prefer operations that can use less-loaded machines
    # Use soft-min to smoothly identify the best (least loaded) machine option
    if num_machines > 1:
        # Adaptive temperature for soft-min based on load spread
        load_range = float(np.max(loads) - np.min(loads))
        load_mean = float(np.mean(loads)) + 1e-9
        temp = 1.0 * (1.0 + 0.3 * load_range / load_mean)
        
        scaled = loads / temp
        max_scaled = np.max(scaled)
        shifted = scaled - max_scaled
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(shifted)))
        soft_min_load = temp * (np.log(num_machines) - log_sum_exp)
        
        # Normalize: lower soft_min_load = less congested = better
        # Convert to a bonus: more bonus for less congestion
        max_possible_load = float(np.max(loads)) + 1e-9
        congestion_bonus = 0.5 * (1.0 - soft_min_load / max_possible_load)
    else:
        congestion_bonus = 0.0
    
    # Tertiary: Job urgency - fewer remaining ops = closer to completion
    remaining_ops = op.get("remaining_ops", 0)
    urgency_bonus = 0.2 / (remaining_ops + 1.0)
    
    # Flexibility: more machine options is slightly better
    flexibility_bonus = 0.1 * np.log1p(num_machines)
    
    # Combine: LPT dominates, congestion and urgency break ties
    score = 1.0 * max_dur + congestion_bonus + urgency_bonus + flexibility_bonus
    
    return score


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Returns: index of the ready operation to start.
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a fast rule
    if not _ensure_deadline():
        # Fast fallback: LPT (longest processing time)
        best_idx = 0
        best_dur = -1.0
        for i, op in enumerate(ready_operations):
            durations = list(op.get("machine_options", {}).values())
            if durations:
                max_dur = max(durations)
                if max_dur > best_dur:
                    best_dur = max_dur
                    best_idx = i
        return best_idx

    # Build machine load estimates from ready operations
    # This is a lightweight proxy for actual machine congestion
    machine_loads = {}
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in machine_loads:
                machine_loads[machine] = 0.0
            # Weight by duration: longer ops create more congestion
            machine_loads[machine] += duration

    # Compute scores for all ready operations
    scores = np.array([_compute_op_score(op, time, machine_loads) for op in ready_operations], dtype=float)

    # Add tiny random tiebreaker to avoid systematic tie-breaking
    if len(ready_operations) > 1:
        scores += np.random.uniform(0, 1e-6, size=len(scores))

    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
