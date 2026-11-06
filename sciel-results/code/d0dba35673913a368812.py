"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    """
    Score each assembly task using a 'bottleneck slack' heuristic.
    
    The heuristic prioritizes tasks with the smallest ratio of 
    (cycle_time - time_i) to the number of immediate successors.
    This favors tasks that consume the most available station capacity 
    relative to the branching complexity they unlock.
    
    Since the station filling algorithm picks the HIGHEST priority eligible task,
    we want tasks that are "most critical" to have the HIGHEST score.
    
    The raw ratio R_i = (cycle_time - time_i) / max(1, num_successors_i)
    - Small R_i means: little slack and/or many successors -> high priority
    - Large R_i means: lots of slack and/or few successors -> low priority
    
    So we invert: score_i = 1 / (R_i + epsilon) or use negative R_i.
    Since we want highest priority = highest score, and small R_i is good:
    score_i = 1 / (R_i + small_constant)
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists
    successors = defaultdict(list)
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
    
    # Compute number of immediate successors for each task
    num_successors = np.zeros(n, dtype=float)
    for i in range(n):
        num_successors[i] = len(successors[i])
    
    # Ensure at least 1 to avoid division by zero
    num_successors_safe = np.maximum(num_successors, 1.0)
    
    # Compute slack: cycle_time - time_i
    # Clip to avoid negative values (in case time_i > cycle_time, though unlikely)
    slack = np.maximum(cycle_time - times, 1e-6)
    
    # Compute the bottleneck slack ratio
    ratio = slack / num_successors_safe
    
    # Invert so that smaller ratio -> higher score
    # Add a small constant to avoid division by zero
    scores = 1.0 / (ratio + 1e-6)
    
    return scores
# EVOLVE-BLOCK-END
