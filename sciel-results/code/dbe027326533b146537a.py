"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Water-Filling Load-Balancing Heuristic
    # Step 1: Compute current load on each machine from all ready operations
    machine_loads = defaultdict(float)
    for op in ready_operations:
        for machine, duration in op["machine_options"].items():
            machine_loads[machine] += duration

    if not machine_loads:
        return 0

    # Step 2: For each candidate operation, simulate assignment to least-loaded
    # compatible machine and score by resulting L-infinity norm plus load spread
    best_index = 0
    best_score = float('inf')

    for i, op in enumerate(ready_operations):
        # Find the least-loaded compatible machine
        min_load_machine = None
        min_load_val = float('inf')

        for machine, duration in op["machine_options"].items():
            load = machine_loads.get(machine, 0)
            if load < min_load_val:
                min_load_val = load
                min_load_machine = machine
            elif load == min_load_val:
                # Tie-breaker: prefer machine where this op has larger duration
                if min_load_machine is not None:
                    prev_dur = op["machine_options"][min_load_machine]
                    if duration > prev_dur:
                        min_load_machine = machine

        if min_load_machine is None:
            continue

        duration = op["machine_options"][min_load_machine]

        # Simulate new loads after starting this op on min_load_machine
        new_loads = dict(machine_loads)
        new_loads[min_load_machine] = min_load_val - duration

        # Compute new L-infinity norm
        new_max_load = max(new_loads.values()) if new_loads else 0

        # Compute load spread (std dev) to encourage water-filling
        loads_list = list(new_loads.values())
        if len(loads_list) > 1:
            mean_load = sum(loads_list) / len(loads_list)
            variance = sum((l - mean_load) ** 2 for l in loads_list) / len(loads_list)
            spread = variance ** 0.5
        else:
            spread = 0

        # Composite score: minimize new max load, with small penalty for imbalance
        score = new_max_load + 0.05 * spread

        if score < best_score:
            best_score = score
            best_index = i

    return int(best_index)
# EVOLVE-BLOCK-END
