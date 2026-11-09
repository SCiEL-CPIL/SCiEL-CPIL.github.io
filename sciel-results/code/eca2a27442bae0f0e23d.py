"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Step 1: Estimate the total potential load for each machine
    # Load[m] = sum of durations of all ready operations that can use machine m
    machine_total_load = {}
    
    # First pass: accumulate load
    for op in ready_operations:
        machine_options = op.get("machine_options", {})
        for machine, duration in machine_options.items():
            if machine not in machine_total_load:
                machine_total_load[machine] = 0.0
            machine_total_load[machine] += duration

    # Step 2: Score each operation based on the ratio of criticality to cumulative load of eligible machines
    best_score = -1
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue

        remaining_work = op.get("remaining_work", 0)
        if remaining_work <= 0:
            remaining_work = 1e-9

        # Calculate cumulative load of eligible machines
        # This penalizes operations that have many options if those options are all busy (MCF heuristic)
        cumulative_load = 0.0
        min_load_machine = None
        min_load_value = float('inf')
        
        for machine, duration in machine_options.items():
            load = machine_total_load.get(machine, duration)
            cumulative_load += load
            
            # Track the least loaded eligible machine for selection
            if load < min_load_value:
                min_load_value = load
                min_load_machine = machine

        if cumulative_load == 0:
            cumulative_load = 1e-9

        score = remaining_work / cumulative_load

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = min_load_machine

    if best_machine is None:
        # Fallback if no valid machine found
        return 0

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
