"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Select the ready operation that minimizes the weighted sum of:
    1. Minimum processing time across eligible machines (shortest processing time first)
    2. Remaining work of the job (to avoid starving jobs with large remaining work)
    3. Number of remaining operations (to prioritize jobs close to completion)
    
    This is a heuristic that balances short processing times with progress on jobs.
    """
    if not ready_operations:
        return 0
    
    best_index = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Get minimum duration across machine options
        machine_options = op["machine_options"]
        if not machine_options:
            min_duration = 0
        else:
            min_duration = min(machine_options.values())
        
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Score: lower is better
        # Prioritize shorter processing times, but also consider job progress
        # Use a weighted combination:
        # - Primary: shortest processing time (to keep machines busy with quick tasks)
        # - Secondary: lower remaining work (to complete jobs sooner)
        # - Tertiary: fewer remaining ops (to finish jobs)
        
        # Normalize-like weighting: 
        # min_duration is the primary driver
        # remaining_work and remaining_ops are tie-breakers/secondary factors
        
        score = min_duration + 0.1 * remaining_work + 0.01 * remaining_ops
        
        if score < best_score:
            best_score = score
            best_index = i
    
    return int(best_index)
# EVOLVE-BLOCK-END
