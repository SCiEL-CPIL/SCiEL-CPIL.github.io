"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_score = -1
    best_index = 0
    best_machine = None

    for i, op in enumerate(ready_operations):
        # Get remaining work for the job (proxy for critical path length)
        job_remaining_work = op.get("remaining_work", 0)
        
        # Get machine options and find the minimum duration
        machine_options = op.get("machine_options", {})
        if not machine_options:
            continue
            
        durations = list(machine_options.values())
        min_duration = min(durations)
        
        # Avoid division by zero
        if min_duration == 0:
            min_duration = 1e-9

        # Calculate score: Ratio of job criticality to operation processing time
        # This prioritizes long jobs that have short current operations
        score = job_remaining_work / min_duration
        
        if score > best_score:
            best_score = score
            best_index = i
            # Find the machine with the minimum duration for this operation
            best_machine = min(machine_options, key=machine_options.get)

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
