"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Global start time for deadline guard
_start_time = None

def dispatch(ready_operations, time):
    global _start_time
    if _start_time is None:
        _start_time = time_module.time()
    
    # Deadline guard: 120s limit, leave 5s margin
    if time_module.time() - _start_time > 115:
        return 0

    if not ready_operations:
        return 0

    # Calculate scores
    scores = []
    for op in ready_operations:
        durations = list(op["machine_options"].values())
        if not durations:
            scores.append(0.0)
            continue
        
        # Use min duration as the effective processing time
        current_duration = min(durations)
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        
        # Composite Score based on observed improvement:
        # Primary: Most Remaining Work (MWRF)
        # Secondary: Shortest Processing Time (SPT) - subtract min_duration
        # Tertiary: Most Remaining Operations (MROF)
        # Weights: Work is dominant. Duration is a tie-breaker.
        score = remaining_work * 100.0 - current_duration * 1.0 + remaining_ops * 0.1
        scores.append(score)
        
    # Return index of max score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
