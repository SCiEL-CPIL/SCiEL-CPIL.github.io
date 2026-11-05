"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule based on Critical Ratio (CR) heuristic.
    
    Heuristic:
    1. Estimate slack for each job: slack = remaining_work - current_time.
       (Assuming remaining_work is the sum of remaining processing times).
    2. Estimate the processing time for the current operation as the minimum
       duration available across its machine options.
    3. Calculate CR = slack / op_duration.
    4. Select the operation with the LOWEST CR.
    
    Rationale:
    - Low slack (urgent jobs) results in low/negative CR, prioritizing them.
    - Among jobs with similar slack, longer operations (larger denominator) 
      result in lower CR, prioritizing them (LPT-like effect to reduce makespan).
    """
    if not ready_operations:
        return 0

    min_cr = float('inf')
    best_idx = 0

    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        
        # Calculate slack as per directive: (sum of remaining durations) - current_time
        # We assume op["remaining_work"] is the sum of remaining durations.
        slack = remaining_work - time
        
        # Get the minimum duration across all machine options
        machine_options = op["machine_options"]
        if machine_options:
            # Use the minimum duration as the expected processing time
            # This assumes the scheduler will pick the fastest available machine
            op_duration = min(machine_options.values())
        else:
            op_duration = 1.0
            
        # Avoid division by zero
        if op_duration <= 0:
            op_duration = 0.1

        # Calculate Critical Ratio
        cr = slack / op_duration
        
        # We want the LOWEST CR
        if cr < min_cr:
            min_cr = cr
            best_idx = i

    return best_idx
# EVOLVE-BLOCK-END
