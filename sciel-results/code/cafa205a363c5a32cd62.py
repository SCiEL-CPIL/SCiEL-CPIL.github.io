"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_score = -np.inf
    best_idx = 0

    for i, op in enumerate(ready_operations):
        # Get the minimum duration among available machines for this operation
        # This represents the "best" machine choice for this operation
        durations = list(op["machine_options"].values())
        if not durations:
            # Fallback if no options, though unlikely
            min_duration = 1.0
        else:
            min_duration = min(durations)
        
        remaining_work = op["remaining_work"]
        
        # Heuristic: Weighted Shortest Processing Time (WSPT)
        # Prioritize operations with high remaining work (critical path)
        # but divide by processing time to favor efficient scheduling
        # This balances "urgency" (remaining work) with "efficiency" (short duration)
        score = remaining_work / (min_duration + 1e-9)
        
        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
