"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_score = -np.inf
    best_idx = 0

    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        remaining_work = op["remaining_work"]
        
        # Approach: Shortest Processing Time (SPT) for machine selection
        # to minimize machine blockage, combined with Most Work Remaining (MWR)
        # to prioritize jobs with more work left (urgency for makespan).
        
        min_dur = np.inf
        for m, dur in machine_options.items():
            if dur < min_dur:
                min_dur = dur
        
        if min_dur == np.inf:
            # No valid options, assign low score
            score = -np.inf
        else:
            # Weighted score: High remaining work is good (urgent), low duration is good (efficient)
            # The weight on remaining_work should be significant to drive completion.
            score = remaining_work - 0.1 * min_dur

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx
# EVOLVE-BLOCK-END
