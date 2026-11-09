"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Pick the ready operation to start next using a Shortest Processing Time (SPT) heuristic.
    
    This approach differs from the job-urgency weighted sum by prioritizing
    the completion of individual operations as quickly as possible.
    
    Heuristic:
    1. Primary: Shortest Processing Time (SPT) - minimize the time the machine is occupied.
    2. Secondary: Most Operations Remaining (MOR) - break ties by prioritizing jobs with more steps left.
    3. Tertiary: Most Work Remaining (MWR) - final tie-breaker.
    
    Rationale:
    SPT is a classic dispatching rule that often reduces makespan by keeping the system
    flowing and preventing long jobs from blocking machines unnecessarily.
    """
    if not ready_operations:
        return 0
    
    # If only one option, return it
    if len(ready_operations) == 1:
        return 0
    
    scores = []
    for op in ready_operations:
        # Get remaining work and operations for the job
        remaining_work = op.get("remaining_work", 0)
        remaining_ops = op.get("remaining_ops", 0)
        
        # Get machine options and their durations
        machine_options = op.get("machine_options", {})
        if machine_options:
            durations = list(machine_options.values())
            min_duration = min(durations)
        else:
            min_duration = 0
        
        # Score calculation:
        # We want to minimize min_duration.
        # We want to maximize remaining_ops and remaining_work.
        
        # To use argmax, we invert the primary metric (min_duration).
        # Scale factors are chosen to make SPT the dominant factor.
        
        # If min_duration is 0, it's free/instant, prioritize it.
        # We use a large negative weight for duration to ensure SPT dominates.
        
        score = (
            -1000.0 * min_duration +      # Primary: Shortest Processing Time
            10.0 * remaining_ops +         # Secondary: Most Operations Remaining
            1.0 * remaining_work           # Tertiary: Most Work Remaining
        )
        
        scores.append(score)
    
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
