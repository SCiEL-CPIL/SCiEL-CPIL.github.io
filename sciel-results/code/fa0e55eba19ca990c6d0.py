"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

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
        # pick machine with smallest duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        return (0, best_m)

    # 1. Identify the "best" machine for each operation (minimum duration)
    # and calculate the "pressure" on each machine.
    # Pressure is defined as the number of operations for which this machine
    # is the minimum-duration option. This indicates contention.
    
    best_machines = [None] * n
    min_durs = [0.0] * n
    machine_pressure = {}

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Find machine with min duration
        best_m = min(opts, key=lambda m: (opts[m], m))
        best_machines[i] = best_m
        min_durs[i] = opts[best_m]
        
        if best_m not in machine_pressure:
            machine_pressure[best_m] = 0
        machine_pressure[best_m] += 1

    # Deadline guard: fall back to a simple deterministic choice.
    if not _elapsed_ok():
        idx = max(range(n), key=lambda i: ready_operations[i]["remaining_work"])
        return (idx, best_machines[idx])

    # 2. Calculate an "effective cost" for each operation.
    # Effective cost = min over all machine options of (duration + weight * pressure[machine])
    # This allows us to consider switching to a less contended machine even if
    # its duration is slightly longer, provided the pressure is low enough.
    
    WEIGHT = 1.0  # Weight for machine pressure in effective cost calculation
    
    effective_costs = [0.0] * n
    best_effective_machines = [None] * n

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        min_cost = float('inf')
        chosen_m = None
        for m, dur in opts.items():
            press = machine_pressure.get(m, 0)
            cost = dur + WEIGHT * press
            if cost < min_cost:
                min_cost = cost
                chosen_m = m
        effective_costs[i] = min_cost
        best_effective_machines[i] = chosen_m

    # 3. Score and sort operations.
    # Primary: -remaining_work (Higher work = Higher priority)
    # Secondary: effective_cost (Lower cost = Higher priority, balances load)
    # Tertiary: remaining_ops (Fewer ops = Higher priority, completes jobs)
    # Quaternary: job (Lower ID = Higher priority, determinism)
    
    def score_op(i):
        op = ready_operations[i]
        primary = -op["remaining_work"]
        secondary = effective_costs[i]
        tertiary = op["remaining_ops"]
        quaternary = op["job"]
        return (primary, secondary, tertiary, quaternary)

    order = sorted(range(n), key=score_op)
    idx = order[0]
    
    # Return the index and the machine that minimizes the effective cost for this op
    return (idx, best_effective_machines[idx])

# EVOLVE-BLOCK-END
