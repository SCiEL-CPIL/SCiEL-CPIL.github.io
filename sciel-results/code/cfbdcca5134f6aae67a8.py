"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 140.0  # Safety margin before 150s kill
_START_TIME = _time.time()


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Uses a hybrid efficiency metric: Remaining Work / Current Min Duration.
    This favors jobs with short initial steps relative to their total remaining load,
    reducing machine blocking and improving flow.
    
    Also includes:
    - Machine contention consideration via number of machine options
    - Time-aware deadline guard for safety
    """
    if not ready_operations:
        return 0
    
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        return 0
    
    n = len(ready_operations)
    
    # Compute scores for each ready operation
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        # Get machine options and find the minimum duration (SPT metric)
        machine_options = op["machine_options"]
        durations = list(machine_options.values())
        min_duration = min(durations)
        
        # Remaining work for this job
        remaining_work = op["remaining_work"]
        
        # Number of remaining operations
        remaining_ops = op["remaining_ops"]
        
        # Hybrid efficiency metric: remaining_work / min_duration
        # This prioritizes operations where the current step is short relative to total work
        if min_duration > 0:
            efficiency = remaining_work / min_duration
        else:
            efficiency = remaining_work + 1000  # Avoid division by zero, treat as very high priority
        
        # Machine contention: fewer options means higher contention risk
        num_machines = len(machine_options)
        contention_penalty = -0.05 * num_machines
        
        # Composite score:
        # - Primary: efficiency (higher is better) - short steps relative to remaining work
        # - Secondary: remaining work (higher is better) - prioritize jobs with more work
        # - Tertiary: fewer remaining ops (lower is better) - break ties
        # - Contention penalty for operations with many machine choices
        
        scores[i] = 1.0 * efficiency + 0.1 * remaining_work - 0.05 * remaining_ops + contention_penalty
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
