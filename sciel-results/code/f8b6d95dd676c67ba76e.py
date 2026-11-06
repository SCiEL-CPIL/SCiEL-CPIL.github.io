"""Open-Shop Scheduling: Load-Balancing Water-Filling dispatch rule."""
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

    # Pre-calculate metrics for each ready operation
    min_durs = []
    best_machines = []
    remaining_works = []
    remaining_ops = []
    job_ids = []

    for op in ready_operations:
        opts = op["machine_options"]
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines.append(best_m)
        min_durs.append(opts[best_m])
        remaining_works.append(op["remaining_work"])
        remaining_ops.append(op["remaining_ops"])
        job_ids.append(op["job"])

    if not _elapsed_ok():
        idx = int(np.argmax(remaining_works))
        return (idx, best_machines[idx])

    # Compute machine loads for load-balanced machine selection
    all_machines = set()
    for op in ready_operations:
        all_machines.update(op["machine_options"].keys())

    if not all_machines:
        return (0, best_machines[0])

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

    # Load-Balancing Water-Filling scoring:
    # For each candidate operation, simulate assignment to the least-loaded compatible machine
    # and score by the resulting increase in the maximum machine load (L-infinity norm).
    # Prioritize operations that minimally skew the global workload distribution.
    # Secondary: CPM proxy (higher remaining work = higher priority)
    # Tertiary: SPT (shorter min processing time)
    # Quaternary: finish jobs sooner (fewer remaining ops)

    scores = []
    for i in range(n):
        op = ready_operations[i]
        opts = op["machine_options"]

        # Find the least-loaded compatible machine
        min_load_machine = min(opts.keys(), key=lambda m: (machine_loads.get(m, 0.0), m))
        dur = opts[min_load_machine]
        new_load = machine_loads.get(min_load_machine, 0.0) + dur

        # Compute the new maximum load if this operation is assigned
        # (only this machine's load changes)
        if new_load > current_max_load:
            load_increase = new_load - current_max_load
        else:
            # Check if any other machine has a higher load
            # Since we're only increasing one machine, the max can only stay or increase
            load_increase = 0.0

        # Also compute a smooth surrogate: soft-max over all machine loads after assignment
        # This captures the overall load distribution, not just the peak
        # For efficiency, we use a weighted combination

        # CPM proxy: prioritize jobs with most remaining work
        # SPT: shorter min processing time as tie-breaker
        # Last operation of job gets slight priority

        last_bonus = 0 if remaining_ops[i] == 1 else 1

        # Composite score (lower is better):
        # 1. L-infinity load increase (primary: water-filling)
        # 2. -remaining_work (CPM: higher remaining work = higher priority)
        # 3. min_dur (SPT tie-break)
        # 4. load increase relative to mean (smooth surrogate)
        # 5. last_bonus (finish jobs sooner)

        mean_load_increase = max(0.0, new_load - current_mean_load)

        score = (
            load_increase,           # L-infinity: minimize peak load increase
            -remaining_works[i],     # CPM: higher remaining work first
            min_durs[i],             # SPT tie-break
            mean_load_increase,      # smooth surrogate for load balance
            last_bonus,              # finish jobs sooner
        )
        scores.append(score)

    idx = min(range(n), key=lambda i: scores[i])

    # For the selected operation, pick the machine that minimizes the resulting max load
    # (smooth surrogate for hard constraint satisfaction)
    opts = ready_operations[idx]["machine_options"]
    best_m = min(opts.keys(), key=lambda m: (
        max(machine_loads.get(m, 0.0) + opts[m], current_max_load) - current_max_load,
        machine_loads.get(m, 0.0) + opts[m],
        m
    ))
    return (idx, best_m)
# EVOLVE-BLOCK-END
