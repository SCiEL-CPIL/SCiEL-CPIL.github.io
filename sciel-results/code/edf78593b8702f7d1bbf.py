"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Longest Processing Time (LPT) approximation for makespan minimization.
    Select the ready operation with the maximum minimum processing time (max min_duration)
    to balance machine loads early and prevent late-stage bottlenecks.
    
    This prioritizes long, inflexible tasks while they are still flexible in machine choice,
    reducing the risk of creating bottlenecks later in the schedule.
    """
    if not ready_operations:
        return 0
    
    # For each ready operation, compute the minimum duration across its machine options
    # This represents the "inflexibility" or "criticality" of the operation
    min_durations = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if machine_options:
            min_dur = min(machine_options.values())
        else:
            min_dur = 0
        min_durations.append(min_dur)
    
    # Select the operation with the maximum minimum duration (LPT-like)
    # This prioritizes operations that are "long" in their best-case scenario
    best_idx = int(np.argmax(min_durations))
    
    return best_idx
# EVOLVE-BLOCK-END
