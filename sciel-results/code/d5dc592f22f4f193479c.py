"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy list scheduler.
    Stations are filled with the highest-priority eligible task that fits the cycle time.
    We want to minimize the mean number of stations over 5 instances.
    
    Strategy: Combine multiple heuristics into a single priority score.
    1. Longest processing time (LPT) - favors long tasks to pack them tightly.
    2. Longest path to end (critical path) - favors tasks that unlock long chains.
    3. Weighted precedence / successor count - favors tasks with many successors (branching).
    4. Bottleneck slack: (cycle_time - time_i) / num_successors - favors tasks that consume
       a lot of capacity relative to the branching they unlock.
    
    We combine these into a single score. The greedy scheduler picks the highest score
    among eligible tasks that fit.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]  # list of lists: predecessors[i] = list of task indices that must precede i
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # 1. Longest processing time
    lpt = times.copy()
    
    # 2. Longest path from each task to the end (critical path)
    # longest_path[i] = max over successors j of (longest_path[j] + times[j]), or 0 if no successors
    # We compute this via topological order (reverse)
    # Topological sort
    in_degree = [len(predecessors[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't have all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use LPT
        return lpt
    
    # Compute longest path to end
    longest_path = np.zeros(n, dtype=float)
    # Process in reverse topological order
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in successors[u]:
            max_succ = max(max_succ, times[v] + longest_path[v])
        longest_path[u] = max_succ
    
    # 3. Number of immediate successors
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # 4. Bottleneck slack: (cycle_time - time_i) / (num_successors + 1)
    # Lower slack means the task consumes more capacity relative to branching it unlocks
    # We want to prioritize tasks with LOW slack, so we invert: priority = 1/(slack + eps)
    eps = 1e-9
    slack = (cycle_time - times) / (num_successors + 1.0)
    # Invert so that lower slack -> higher priority
    bottleneck_score = 1.0 / (slack + eps)
    
    # Normalize each component to [0, 1] range for fair combination
    def normalize(arr):
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    lpt_norm = normalize(lpt)
    lp_norm = normalize(longest_path)
    # For num_successors, more successors = more branching = higher priority
    succ_norm = normalize(num_successors)
    # For bottleneck_score, higher = lower slack = higher priority
    bottleneck_norm = normalize(bottleneck_score)
    
    # Combine with weights
    # LPT is a strong baseline; critical path adds nuance; successors add branching awareness
    # Bottleneck slack adds capacity-utilization perspective
    w_lpt = 0.35
    w_lp = 0.30
    w_succ = 0.15
    w_bottleneck = 0.20
    
    priority_scores = (w_lpt * lpt_norm + 
                       w_lp * lp_norm + 
                       w_succ * succ_norm + 
                       w_bottleneck * bottleneck_norm)
    
    # Add a small tie-breaker based on original time to ensure deterministic ordering
    priority_scores = priority_scores * 1.0 + lpt * 1e-6
    
    return priority_scores
# EVOLVE-BLOCK-END
