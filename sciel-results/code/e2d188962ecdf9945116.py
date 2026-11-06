"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    best_idx = -1
    best_min_dur = float('inf')
    best_machine = None
    best_remaining_ops = float('inf')
    
    for i, op in enumerate(ready_operations):
        # Find the machine with the shortest processing time for this operation
        min_dur = float('inf')
        min_machine = None
        for m, d in op["machine_options"].items():
            if d < min_dur:
                min_dur = d
                min_machine = m
        
        # Primary criterion: Shortest Processing Time (SPT)
        # Secondary criterion: Fewest remaining operations (to complete jobs early)
        # Tertiary criterion: Most remaining work (arbitrary tie-breaker, or use index)
        
        if min_dur < best_min_dur:
            best_min_dur = min_dur
            best_machine = min_machine
            best_remaining_ops = op["remaining_ops"]
            best_idx = i
        elif min_dur == best_min_dur:
            if op["remaining_ops"] < best_remaining_ops:
                best_remaining_ops = op["remaining_ops"]
                best_machine = min_machine
                best_idx = i
            elif op["remaining_ops"] == best_remaining_ops:
                # Tie-break by index to be deterministic
                if i < best_idx:
                    best_machine = min_machine
                    best_idx = i

    if best_machine is not None:
        return (best_idx, best_machine)
    else:
        return best_idx
# EVOLVE-BLOCK-END
