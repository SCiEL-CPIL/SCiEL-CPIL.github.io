"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START = _time.time()
_DEADLINE = 150.0 - 5.0  # safety margin


def _elapsed_ok():
    return (_time.time() - _START) < _DEADLINE


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # Pre-calculate metrics
    min_durs = []
    best_machines = []
    is_last_ops = []
    remaining_works = []

    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        is_last_ops.append(op["remaining_ops"] == 1)
        remaining_works.append(op["remaining_work"])

    if not _elapsed_ok():
        idx = int(np.argmax(remaining_works))
        return (idx, best_machines[idx])

    # Collect all machines
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())

    if not all_machines:
        return (0, best_machines[0])

    # Estimate machine loads with ratio-based weighting:
    # For each op, weight each machine by (best_dur / dur_on_machine),
    # so the best machine gets weight 1.0 and worse machines get < 1.0.
    # This naturally emphasizes the preferred machine while still
    # accounting for alternatives.
    machine_loads = {m: 0.0 for m in all_machines}
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        best_dur = min_durs[i]
        for m, dur in opts.items():
            if dur > 0:
                weight = best_dur / dur
            else:
                weight = 1.0
            machine_loads[m] += dur * weight

    load_values = np.array(list(machine_loads.values()), dtype=float)
    current_max_load = float(np.max(load_values)) if len(load_values) > 0 else 0.0
    current_mean_load = float(np.mean(load_values)) if len(load_values) > 0 else 0.0
    n_machines = len(all_machines)

    # Score each candidate:
    scores = []
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]

        # Assign to least-loaded compatible machine
        min_load_machine = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        dur = opts[min_load_machine]
        new_load = machine_loads.get(min_load_machine, 0.0) + dur

        # Primary: increase in max load
        new_max_load = max(current_max_load, new_load)
        load_increase = new_max_load - current_max_load

        # Secondary: deviation of the assigned machine from mean
        deviation = abs(new_load - current_mean_load)

        # Tertiary: balance metric — how uneven are loads after this assignment?
        # Approximate: if the assigned machine becomes the new max, the spread increases.
        # Use: max(0, new_load - current_mean_load) as a proxy for imbalance
        imbalance = max(0.0, new_load - current_mean_load)

        # Tie-breakers
        last_bonus = 0 if is_last_ops[i] else 1
        spt = min_durs[i]
        lpt = -remaining_works[i]

        score = (
            load_increase,
            imbalance,
            deviation,
            last_bonus,
            spt,
            lpt,
        )
        scores.append(score)

    idx = min(range(n), key=lambda i: scores[i])
    best_m = best_machines[idx]
    return (idx, best_m)
# EVOLVE-BLOCK-END
