"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Improved heuristic: Combine Most Remaining Work (MRW) with
    # Minimum Remaining Processing Time (MRPT) for the current operation,
    # and prefer machines with shorter processing times when there's a tie
    # or when the operation has flexible machine options.
    
    best_idx = 0
    best_score = -float('inf')
    
    for i, op in enumerate(ready_operations):
        # Use remaining_work if available, otherwise fall back to min duration
        work = op.get("remaining_work", 0)
        if work == 0:
            options = op.get("machine_options", {})
            if options:
                work = min(options.values())
            else:
                work = 0
        
        # Get the minimum processing time for this operation
        options = op.get("machine_options", {})
        if options:
            min_proc = min(options.values())
            # Number of machine options (flexibility)
            num_options = len(options)
        else:
            min_proc = 0
            num_options = 1
        
        # Score: prioritize high remaining work, but also consider
        # short processing times for the current operation to avoid
        # blocking machines. We use a weighted combination.
        # Higher remaining work = higher priority
        # Lower min_proc = slightly higher priority (to free machines quickly)
        # More options = slightly higher priority (more flexibility)
        
        score = work * 1.0 + (1.0 / (min_proc + 1)) * 0.1 + num_options * 0.05
        
        if score > best_score:
            best_score = score
            best_idx = i
            
    return best_idx
# EVOLVE-BLOCK-END
