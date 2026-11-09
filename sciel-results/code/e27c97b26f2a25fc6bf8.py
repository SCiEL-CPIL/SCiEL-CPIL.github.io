"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def priority(instance):
    """
    Compute priority scores for assembly tasks using Critical Path Density heuristic.
    
    Priority is based on the ratio of task time to slack, where slack is the
    difference between cycle time and the task's own time plus its longest
    downstream path. This prioritizes tasks that are both long and have little
    scheduling flexibility.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor graph from predecessors
    successors = [[] for _ in range(n)]
    for i, preds in enumerate(predecessors):
        for p in preds:
            successors[p].append(i)
    
    # Compute longest downstream path using topological order (reverse)
    # downstream[i] = max(time[i] + max(downstream[j] for j in successors[i]), time[i])
    downstream = np.zeros(n, dtype=float)
    
    # Get topological order using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i, preds in enumerate(predecessors):
        for p in preds:
            in_degree[i] += 1
    
    # Actually, we need to process in reverse topological order
    # Let's compute in-degree for successors (out-degree in original)
    out_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for j in successors[i]:
            out_degree[i] += 1
    
    # Process nodes with no successors first (sinks), then work backwards
    # Use a queue of nodes with out_degree 0
    from collections import deque
    queue = deque()
    for i in range(n):
        if out_degree[i] == 0:
            queue.append(i)
    
    processed = np.zeros(n, dtype=bool)
    # downstream[i] will be the longest path starting at i (including time[i])
    # For sinks: downstream[i] = time[i]
    # For others: downstream[i] = time[i] + max(downstream[j] for j in successors[i])
    
    # We need to process in reverse topological order
    # Let's build the reverse: process nodes whose all successors have been processed
    # Track how many successors have been processed for each node
    succ_processed = np.zeros(n, dtype=int)
    
    while queue:
        i = queue.popleft()
        if processed[i]:
            continue
        processed[i] = True
        
        # downstream[i] = time[i] + max(downstream[j] for j in successors[i] that are processed)
        # But we process in order where successors are already done
        max_succ = 0.0
        for j in successors[i]:
            if downstream[j] > max_succ:
                max_succ = downstream[j]
        
        downstream[i] = times[i] + max_succ
        
        # Decrement count for predecessors of i
        for p in predecessors[i]:
            succ_processed[p] += 1
            if succ_processed[p] == len(successors[p]):
                queue.append(p)
    
    # Some nodes might not be processed if there are cycles (shouldn't happen in valid input)
    for i in range(n):
        if not processed[i]:
            downstream[i] = times[i]
    
    # Compute slack: slack[i] = cycle_time - downstream[i]
    # If slack <= 0, the task is on the critical path and should have very high priority
    slack = cycle_time - downstream
    
    # Compute priority: time / slack, with handling for slack <= 0
    # For slack <= 0, use a large value
    # For slack > 0, use time / slack
    
    # Use a safe division to avoid issues
    eps = 1e-9
    safe_slack = np.where(slack > eps, slack, eps)
    
    # Base priority: time / slack
    priority_scores = times / safe_slack
    
    # Boost priority for tasks with slack <= 0 (critical path tasks)
    critical_mask = slack <= eps
    priority_scores[critical_mask] = np.max(priority_scores) + times[critical_mask]
    
    # Add a small tiebreaker: prefer tasks with more successors (they constrain more)
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    max_succ = np.max(succ_count) if max_succ > 0 else 1.0
    priority_scores += 0.01 * succ_count / max_succ
    
    return priority_scores
# EVOLVE-BLOCK-END
