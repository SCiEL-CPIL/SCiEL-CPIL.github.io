"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard: evaluator kills after 150s wall-clock; use 120s as safe margin
_DEADLINE = 120.0
_start_time = _time.monotonic()


def _elapsed():
    return _time.monotonic() - _start_time


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Strategy: Composite heuristic that balances:
    1. Most remaining work (primary urgency)
    2. Most remaining operations (structural urgency - critical path)
    3. Shortest minimum duration (unlock next ops sooner)
    4. Soft-min bottleneck awareness (avoid overloading critical machines)
    
    Returns the index of the ready operation to start.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Deadline guard
    if _elapsed() > _DEADLINE:
        return 0

    # Extract features for each operation
    remaining_works = []
    remaining_ops = []
    min_durations = []
    max_durations = []
    num_machines = []
    
    for op in ready_operations:
        work = op["remaining_work"]
        rem_ops = op.get("remaining_ops", 0)
        durations = list(op["machine_options"].values())
        if not durations:
            remaining_works.append(0)
            remaining_ops.append(0)
            min_durations.append(float('inf'))
            max_durations.append(0)
            num_machines.append(1)
        else:
            remaining_works.append(work)
            remaining_ops.append(rem_ops)
            min_durations.append(min(durations))
            max_durations.append(max(durations))
            num_machines.append(len(durations))
    
    remaining_works = np.array(remaining_works, dtype=float)
    remaining_ops = np.array(remaining_ops, dtype=float)
    min_durations = np.array(min_durations, dtype=float)
    max_durations = np.array(max_durations, dtype=float)
    num_machines = np.array(num_machines, dtype=float)
    
    # Handle edge cases
    if np.all(remaining_works == 0):
        return int(np.argmin(min_durations))
    
    # Normalize features to [0, 1] range for weighting
    work_range = remaining_works.max() - remaining_works.min()
    if work_range > 0:
        work_norm = (remaining_works - remaining_works.min()) / work_range
    else:
        work_norm = np.zeros(n)
    
    ops_range = remaining_ops.max() - remaining_ops.min()
    if ops_range > 0:
        ops_norm = (remaining_ops - remaining_ops.min()) / ops_range
    else:
        ops_norm = np.zeros(n)
    
    min_dur_range = min_durations.max() - min_durations.min()
    if min_dur_range > 0:
        min_dur_norm = (min_durations - min_durations.min()) / min_dur_range
    else:
        min_dur_norm = np.zeros(n)
    
    max_dur_range = max_durations.max() - max_durations.min()
    if max_dur_range > 0:
        max_dur_norm = (max_durations - max_durations.min()) / max_dur_range
    else:
        max_dur_norm = np.zeros(n)
    
    # Soft-min approximation: for each op, the "effective bottleneck" is
    # approximated by a log-sum-exp of negative durations (smooth min)
    # We use a simpler smooth surrogate: weighted combo of min and mean
    # to capture both the best-case and average-case duration
    mean_durations = np.array([np.mean(list(op["machine_options"].values())) 
                                if op["machine_options"] else 0 
                                for op in ready_operations], dtype=float)
    
    mean_range = mean_durations.max() - mean_durations.min()
    if mean_range > 0:
        mean_dur_norm = (mean_durations - mean_durations.min()) / mean_range
    else:
        mean_dur_norm = np.zeros(n)
    
    # Composite score: prioritize most work + most remaining ops,
    # then shorter durations to unlock downstream ops
    # Higher score = better
    w_work = 1.0
    w_ops = 0.6
    w_min_dur = 0.12
    w_mean_dur = 0.08
    w_max_dur = 0.04
    
    # For durations, we want shorter to be better, so subtract
    score = (w_work * work_norm + w_ops * ops_norm 
             - w_min_dur * min_dur_norm - w_mean_dur * mean_dur_norm 
             - w_max_dur * max_dur_norm)
    
    # Also consider: operations with more machine options might be more flexible
    # but this is a weak signal, so keep weight small
    if np.max(num_machines) > np.min(num_machines):
        num_mach_range = num_machines.max() - num_machines.min()
        num_mach_norm = (num_machines - num_machines.min()) / num_mach_range
        score += 0.02 * num_mach_norm
    
    # Deadline guard before final selection
    if _elapsed() > _DEADLINE:
        return 0
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(score))
    
    return best_idx
# EVOLVE-BLOCK-END
