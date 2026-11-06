"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Shortest Remaining Work (SRW) with tie-breaking by 
    Shortest Processing Time (SPT).
    
    This rule prioritizes operations from jobs that have the least remaining work.
    The intuition is to complete jobs as soon as possible, which can free up
    resources and reduce the overall makespan by preventing long chains of
    operations from being delayed by shorter, independent jobs.
    
    For each ready operation, we look at the job's remaining_work. We prioritize
    the operation with the minimum remaining_work. In case of ties, we pick the
    one with the shortest processing time (min duration across machines) to
    minimize the immediate time blockage on the machine.
    """
    if not ready_operations:
        return 0

    remaining_works = []
    min_durations = []
    
    for op in ready_operations:
        # Get the minimum duration across all machine options
        if op["machine_options"]:
            min_dur = min(op["machine_options"].values())
        else:
            min_dur = 0
        min_durations.append(min_dur)
        remaining_works.append(op["remaining_work"])

    remaining_works = np.array(remaining_works)
    min_durations = np.array(min_durations)

    # Primary key: Minimize remaining_work
    # Secondary key: Minimize min_duration
    
    min_rwork = np.min(remaining_works)
    # Mask of operations that have the minimum remaining work
    mask = remaining_works == min_rwork
    
    if np.sum(mask) == 1:
        return int(np.where(mask)[0][0])
    
    # If multiple, pick the one with min duration among the masked ones
    candidates = np.where(mask)[0]
    
    # Among candidates, find the one with min duration
    best_idx = candidates[np.argmin(min_durations[candidates])]
    
    return int(best_idx)

# EVOLVE-BLOCK-END
