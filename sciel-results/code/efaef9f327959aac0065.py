"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy station-filling heuristic.
    
    Strategy: "bottleneck slack" heuristic that prioritizes tasks with the
    smallest ratio of (cycle_time - time_i) to the number of immediate successors,
    effectively favoring tasks that consume the most available station capacity
    relative to the branching complexity they unlock.
    
    We combine this with:
    - Longest processing time (LPT) as a base
    - Successor count penalty (tasks with fewer successors are more urgent)
    - Predecessor depth (tasks deeper in the chain have less flexibility)
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor counts and depths
    # predecessors[i] is a list of predecessor task indices for task i
    succ_count = np.zeros(n, dtype=float)
    depth = np.zeros(n, dtype=float)
    
    for i in range(n):
        preds = predecessors[i]
        if preds is not None and len(preds) > 0:
            for p in preds:
                succ_count[p] += 1
            # depth[i] = 1 + max(depth[p] for p in preds)
            max_pred_depth = max(depth[p] for p in preds)
            depth[i] = 1.0 + max_pred_depth
        else:
            depth[i] = 0.0
    
    # Compute priority score
    # Base: longer tasks first
    base = times.copy()
    
    # Slack term: (cycle_time - time_i) / (succ_count_i + 1)
    # Smaller slack ratio means the task is more "tight" relative to its branching
    # We want to prioritize tasks with SMALL slack ratio, so we invert or subtract
    slack = (cycle_time - times) / (succ_count + 1.0)
    
    # Depth bonus: deeper tasks have less flexibility, prioritize them
    depth_bonus = depth * 0.5
    
    # Combine: high base + low slack + high depth_bonus = high priority
    # Normalize slack to be in a reasonable range
    # Use negative slack so smaller slack gives higher score
    priority_scores = base - 0.5 * slack + depth_bonus
    
    return priority_scores
# EVOLVE-BLOCK-END
