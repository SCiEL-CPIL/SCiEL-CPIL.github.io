"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy: Shortest Processing Time (SPT) on the selected machine,
    with a tie-breaker of Longest Remaining Processing Time (LRPT) for the job.
    
    Rationale:
    - SPT helps prevent long blocks on machines, keeping the system flowing.
    - LRPT tie-breaker ensures that jobs with significant remaining work are 
      prioritized when immediate durations are similar, preventing long tails.
    """
    if not ready_operations:
        return 0

    best_idx = 0
    best_min_duration = float('inf')
    best_remaining_work = -1

    for i, op in enumerate(ready_operations):
        # Calculate the minimum duration this operation can take
        # assuming it is assigned to the fastest available machine
        min_duration = min(op["machine_options"].values())
        remaining_work = op["remaining_work"]

        # Primary key: Shortest minimum duration (SPT)
        # Secondary key: Most remaining work (LRPT)
        if (min_duration < best_min_duration or
            (min_duration == best_min_duration and remaining_work > best_remaining_work)):
            best_min_duration = min_duration
            best_remaining_work = remaining_work
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
