"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_dispatch_start_time = _time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

# Track machine availability (estimated completion time) for lookahead
_mach_avail = {}
# Track per-job cumulative remaining work for critical-path estimation
_job_rem_work = {}


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
            # Pick machine that finishes earliest (availability + duration)
            best_machine = min(
                machine_options,
                key=lambda m: _mach_avail.get(m, 0.0) + machine_options[m]
            )
            _mach_avail[best_machine] = max(_mach_avail.get(best_machine, 0.0), time) + machine_options[best_machine]
            return (0, best_machine)
        return 0

    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # For each operation, compute best (min) duration and soft-min completion time
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)

    # Soft-min over machine completion times: -log(sum(exp(-d/tau))) * tau
    # This gives a smooth estimate of when the op could finish on its best machine
    tau = 2.0  # temperature for soft-min
    soft_min_completions = np.array([
        -tau * np.log(np.sum(np.exp(-np.array(list(op["machine_options"].values()), dtype=float) / tau)))
        if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)

    # Earliest completion = max(current_time, machine_avail) + duration
    # Use soft-min to approximate best machine completion
    earliest_completions = np.array([
        max(time, min(_mach_avail.get(m, 0.0) for m in op["machine_options"])) + min(op["machine_options"].values())
        if op["machine_options"] else np.inf
        for op in ready_operations
    ], dtype=float)

    # Downstream impact: remaining_work / (remaining_ops + 1) approximates avg op duration
    # Higher ratio means each remaining op is "heavier" → more critical
    avg_op_weight = remaining_works / (remaining_ops + 1.0)

    # Normalize
    max_work = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_ops = np.max(remaining_ops) if np.max(remaining_ops) > 0 else 1.0
    max_earliest = np.max(earliest_completions) if np.any(earliest_completions < np.inf) else 1.0
    min_earliest = np.min(earliest_completions) if np.any(earliest_completions < np.inf) else 0.0
    range_earliest = max_earliest - min_earliest if max_earliest > min_earliest else 1.0

    # Normalize each feature to [0, 1]
    norm_work = remaining_works / (max_work + 1e-9)
    norm_ops = remaining_ops / (max_ops + 1e-9)
    norm_speed_min = (1.0 / min_durations) / (np.max(1.0 / min_durations) + 1e-9) if np.any(min_durations > 0) else np.zeros(n)
    # Invert earliest completion: earlier completion is better
    norm_earliest = 1.0 - (earliest_completions - min_earliest) / (range_earliest + 1e-9)
    norm_earliest = np.clip(norm_earliest, 0.0, 1.0)

    # Weighted scoring:
    # - Urgency (remaining work): critical path length
    # - Depth (remaining ops): number of future ops
    # - Speed (fastest machine): prefer short processing times
    # - Earliest completion: prefer ops that can finish soonest (unblocks downstream)
    w_work = 1.0
    w_ops = 0.3
    w_speed_min = 0.35
    w_earliest = 0.5

    scores = (w_work * norm_work +
              w_ops * norm_ops +
              w_speed_min * norm_speed_min +
              w_earliest * norm_earliest)

    best_idx = int(np.argmax(scores))

    # Choose the machine that minimizes completion time for the selected operation
    machine_options = ready_operations[best_idx]["machine_options"]
    if machine_options:
        best_machine = min(
            machine_options,
            key=lambda m: _mach_avail.get(m, 0.0) + machine_options[m]
        )
        # Update availability
        _mach_avail[best_machine] = max(_mach_avail.get(best_machine, 0.0), time) + machine_options[best_machine]
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
