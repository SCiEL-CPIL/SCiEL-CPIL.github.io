"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

# Track machine load to prefer less-busy machines
_machine_load = {}
_job_remaining_work = {}
_job_remaining_ops = {}


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    elapsed = _time.time() - _dispatch_start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: simple urgency-based dispatch
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        machine_options = op["machine_options"]
        if machine_options:
            # Pick the least-loaded eligible machine
            best_machine = min(machine_options, key=lambda m: _machine_load.get(m, 0))
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, compute the best (min) duration and the machine-specific durations
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)
    
    # Average duration across all machine options (efficiency signal)
    avg_durations = np.array([
        np.mean(list(op["machine_options"].values())) if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)

    # Normalize
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_ops = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    max_inv_min_dur = np.max(1.0 / min_durations) if np.any(min_durations > 0) else 1.0
    max_inv_avg_dur = np.max(1.0 / avg_durations) if np.any(avg_durations > 0) else 1.0

    # Normalize each feature to [0, 1]
    norm_work = remaining_works / (max_work + 1e-9)
    norm_ops = remaining_ops / (max_ops + 1e-9)
    norm_speed_min = (1.0 / min_durations) / (max_inv_min_dur + 1e-9)
    norm_speed_avg = (1.0 / avg_durations) / (max_inv_avg_dur + 1e-9)

    # Weighted scoring:
    # - Urgency (remaining work): critical path length
    # - Depth (remaining ops): number of future ops (more ops = more tail)
    # - Speed (fastest machine): prefer short processing times
    # - Avg speed: prefer operations that are fast on average (flexibility)
    w_work = 1.0
    w_ops = 0.35
    w_speed_min = 0.45
    w_speed_avg = 0.15

    scores = (w_work * norm_work +
              w_ops * norm_ops +
              w_speed_min * norm_speed_min +
              w_speed_avg * norm_speed_avg)

    best_idx = int(np.argmax(scores))

    # Choose the least-loaded eligible machine for the selected operation
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        # Pick machine with lowest current load; tie-break by shortest duration
        best_machine = min(
            machine_options,
            key=lambda m: (_machine_load.get(m, 0), machine_options[m])
        )
        # Update load tracking
        _machine_load[best_machine] = _machine_load.get(best_machine, 0) + machine_options[best_machine]
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
