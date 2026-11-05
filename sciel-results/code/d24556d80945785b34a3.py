"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math


def dispatch(ready_operations, time):
    """
    Dispatch rule using a composite heuristic that balances:
    1. Job urgency (more remaining work = more urgent)
    2. Operation criticality (shorter min processing time = less blocking)
    3. Job completion proximity (fewer remaining ops = closer to done)
    
    We use a weighted score:
    - Primary: remaining_work (total work left for the job)
    - Secondary: inverse of min machine duration (prefer shorter ops)
    - Tertiary: inverse of remaining_ops (prefer jobs closer to completion)
    
    The idea is to start operations from jobs that have the most total work remaining,
    but among those, prefer operations that can be completed quickly (to free up machines),
    and prefer jobs that are closer to completion (to avoid long tails).
    """
    if not ready_operations:
        return 0
    
    best_idx = 0
    best_score = float('-inf')
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Get the minimum processing time among all eligible machines
        min_duration = min(machine_options.values()) if machine_options else 1
        
        # Avoid division by zero
        safe_remaining_ops = max(remaining_ops, 1)
        safe_min_duration = max(min_duration, 0.001)
        
        # Composite score:
        # - remaining_work: jobs with more total work remaining are more urgent
        # - 1/min_duration: shorter operations are less likely to block machines
        # - 1/remaining_ops: jobs closer to completion get a small boost
        #
        # We weight remaining_work heavily since it captures the overall urgency.
        score = remaining_work + 5.0 / safe_min_duration + 2.0 / safe_remaining_ops
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return int(best_idx)
# EVOLVE-BLOCK-END
