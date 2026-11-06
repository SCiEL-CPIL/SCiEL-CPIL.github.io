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
        sum_duration = sum(durations)
        
        # Remaining work for this job
        remaining_work = op["remaining_work"]
        
        # Number of remaining operations
        remaining_ops = op["remaining_ops"]
        
        # Directive-inspired metric: Inverse of (min/sum) = sum/min
        # This prioritizes operations with short min duration (SPT) and high flexibility (many machines)
        if min_duration > 0:
            flexibility_score = sum_duration / min_duration
        else:
            flexibility_score = 1000.0
            
        # Standard urgency: remaining work relative to current step
        if min_duration > 0:
            urgency_score = remaining_work / min_duration
        else:
            urgency_score = remaining_work + 1000.0

        # Composite score:
        # - Primary: Flexibility/Contention Avoidance (sum/min)
        # - Secondary: Urgency (remaining_work / min_duration)
        # - Tertiary: Remaining work (absolute)
        # We weight flexibility heavily as per directive "machine-contention avoidance"
        
        scores[i] = 1.5 * flexibility_score + 0.5 * urgency_score + 0.1 * remaining_work
    
    # Select the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
