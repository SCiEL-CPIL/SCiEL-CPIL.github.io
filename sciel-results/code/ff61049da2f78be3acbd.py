"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0


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
            best_machine = min(machine_options, key=machine_options.get)
            return (0, best_machine)
        return 0

    # Extract key features for all ready operations
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ])
    
    # Compute critical path-like estimate: remaining work divided by number of remaining ops
    # This gives an average work per remaining operation
    avg_work_per_op = np.where(remaining_ops > 0, remaining_works / np.maximum(remaining_ops, 1), remaining_works)
    
    # Compute effective duration using soft-min (log-sum-exp) for smooth machine selection
    # This avoids the nonsmooth min and provides a smoother priority signal
    beta = 5.0  # temperature for soft-min; higher = closer to hard min
    soft_min_durations = np.zeros(n)
    for i in range(n):
        durations = np.array(list(ready_operations[i]["machine_options"].values()), dtype=float)
        if len(durations) > 0:
            log_sum_exp = np.log(np.sum(np.exp(-beta * durations)))
            soft_min_durations[i] = -log_sum_exp / beta
        else:
            soft_min_durations[i] = np.inf
    
    # Use soft-min duration for efficiency scoring (smoother than hard min)
    inv_soft_durations = 1.0 / np.maximum(soft_min_durations, 1e-9)

    # Compute a spread measure: range of machine durations for each op.
    # Ops with more machine flexibility (wider spread) may benefit more from
    # careful machine assignment, but for now we use the min duration as the
    # primary efficiency signal.
    spread_durations = np.zeros(n)
    for i in range(n):
        durations = np.array(list(ready_operations[i]["machine_options"].values()), dtype=float)
        if len(durations) > 0:
            spread_durations[i] = np.max(durations) - np.min(durations)
        else:
            spread_durations[i] = 0.0

    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_avg = np.max(avg_work_per_op) if np.max(avg_work_per_op) > 0 else 1.0
    max_inv_dur = np.max(inv_soft_durations) if np.any(inv_soft_durations > 0) else 1.0

    # Normalize components to [0, 1]
    normalized_urgency = remaining_works / (max_work + 1e-9)
    normalized_avg = avg_work_per_op / (max_avg + 1e-9)
    normalized_efficiency = inv_soft_durations / (max_inv_dur + 1e-9)

    # Weighted score combining:
    # - Urgency: total remaining work (critical path length proxy) — strongest signal
    # - Depth-adjusted urgency: average work per remaining op
    # - Efficiency: how fast the operation can be completed (soft-min over machine options)
    # - Completeness: jobs closer to completion (fewer remaining ops) get a small boost
    w_urgency = 1.2
    w_avg = 0.6
    w_efficiency = 0.4
    w_complete = 0.3

    # Normalize remaining_ops: fewer remaining ops = closer to completion = higher priority
    max_remaining_ops = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    normalized_complete = 1.0 - (remaining_ops / (max_remaining_ops + 1e-9))

    scores = (w_urgency * normalized_urgency + w_avg * normalized_avg +
              w_efficiency * normalized_efficiency + w_complete * normalized_complete)

    best_idx = int(np.argmax(scores))

    # Choose the best machine for the selected operation.
    # Use a composite machine score: prefer shorter duration, but also consider
    # the current simulation time as a proxy for machine load.
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        # Pick the machine with minimum duration; break ties by machine id for determinism
        best_machine = min(machine_options, key=lambda m: (machine_options[m], m))
        return (best_idx, best_machine)
    return best_idx


def _soft_min_value(values, beta=5.0):
    """Compute soft-min using log-sum-exp for numerical stability."""
    if len(values) == 0:
        return np.inf
    arr = np.array(values, dtype=float)
    min_val = np.min(arr)
    # Shift for numerical stability
    shifted = arr - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_val - log_sum / beta
# EVOLVE-BLOCK-END
