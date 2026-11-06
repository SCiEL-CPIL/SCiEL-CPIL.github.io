"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as time_module

# Deadline guard
START_TIME = time_module.time()
DEADLINE = 120.0 # Task limit is 120s, keep margin
TIME_LIMIT = 150.0 # Explicit kill limit mentioned in directive

def dispatch(ready_operations, time):
    # Check deadline
    if time_module.time() > TIME_LIMIT - 10.0:
        # Fallback to simple heuristic to avoid timeout
        return 0

    if not ready_operations:
        return 0

    best_index = 0
    best_machine = None
    best_score = float('inf')
    best_criticality = float('-inf')

    for i, op in enumerate(ready_operations):
        # 1. Determine the best machine for this operation (SPT machine selection)
        # We want the machine that processes this op fastest.
        min_duration = float('inf')
        best_m = None
        for m, d in op["machine_options"].items():
            if d < min_duration:
                min_duration = d
                best_m = m
        
        if min_duration == float('inf'):
            continue

        # 2. Determine priority for this operation
        # Heuristic: SPT (Shortest Processing Time) is primary.
        # Secondary: Criticality (Remaining Work) to break ties or boost urgent jobs.
        
        # We want to minimize min_duration.
        # If min_duration is very small, it's a good candidate.
        
        # Let's use a composite score.
        # To compare operations, we prioritize smaller min_duration.
        # If min_duration is equal, we prioritize larger remaining_work.
        
        # Simple approach:
        # Score = min_duration
        # Tie-breaker = -remaining_work (so larger work is "smaller" in a min-sort sense if we negate, 
        # but it's easier to just track manually)
        
        current_criticality = op.get("remaining_work", 0)
        
        # Logic:
        # If current min_duration < best_score, it's better.
        # If current min_duration == best_score, check criticality.
        # If current criticality > best_criticality, it's better.
        
        if min_duration < best_score:
            best_score = min_duration
            best_index = i
            best_machine = best_m
            best_criticality = current_criticality
        elif min_duration == best_score:
            if current_criticality > best_criticality:
                best_index = i
                best_machine = best_m
                best_criticality = current_criticality

    # Return index and machine
    # The prompt says: "returns index ... (optionally (index, machine))"
    # Providing the machine is better as it allows precise SPT machine selection.
    return (best_index, best_machine)

# EVOLVE-BLOCK-END
