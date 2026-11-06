"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def dispatch(ready_operations, time):
    """
    Dispatch rule: Select the ready operation with the maximum minimum processing time.
    This approximates LPT (Longest Processing Time) to handle bottleneck operations early.
    Ties are broken by maximum remaining_work.
    """
    if not ready_operations:
        return 0

    # Calculate the minimum duration for each ready operation
    # min_duration represents the "best case" processing time, but if this is large,
    # the operation is inherently heavy and should be prioritized to avoid late bottlenecks.
    min_durations = []
    remaining_works = []
    
    for op in ready_operations:
        durations = op["machine_options"].values()
        min_dur = min(durations)
        min_durations.append(min_dur)
        remaining_works.append(op["remaining_work"])
        
    min_durations = np.array(min_durations)
    remaining_works = np.array(remaining_works)
    
    # We want to maximize min_durations, then maximize remaining_works.
    # We can create a composite score or just sort.
    # Using argmax on a tuple isn't directly supported in numpy for multi-criteria without sorting.
    # Let's find the max min_duration, then filter by that, then pick max remaining_work.
    
    max_min_dur = np.max(min_durations)
    candidates = np.where(min_durations == max_min_dur)[0]
    
    if len(candidates) == 1:
        return int(candidates[0])
    
    # Among candidates with the same max min_duration, pick the one with max remaining_work
    candidate_remaining_works = remaining_works[candidates]
    best_candidate_idx_in_candidates = np.argmax(candidate_remaining_works)
    
    return int(candidates[best_candidate_idx_in_candidates])

# EVOLVE-BLOCK-END
