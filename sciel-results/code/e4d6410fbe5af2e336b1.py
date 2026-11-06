"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using a Critical Path (Bottom Level) based heuristic.
    
    This approach differs from the weighted-sum-of-successors method by focusing
    on the longest path (critical path) to the end of the line. This ensures that
    tasks which are part of the bottleneck chains are prioritized, minimizing the
    number of stations required. A small bonus is added for task duration and
    successor count to help with packing and unlocking.
    
    Score = Bottom_Level + 0.1 * Successor_Count + 0.1 * Time
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
        
    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Fallback if cycle detected
    if len(topo_order) < n:
        return times.copy()
    
    # Compute Bottom Level (Longest path from node to any sink)
    # Process in reverse topological order
    bottom_level = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ_bl = 0.0
        for succ in successors[node]:
            if bottom_level[succ] > max_succ_bl:
                max_succ_bl = bottom_level[succ]
        bottom_level[node] = times[node] + max_succ_bl
        
    # Compute Successor Count
    succ_count = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Combine scores
    # Primary: Bottom Level (Critical Path)
    # Secondary: Successor Count (Unlocking power)
    # Tertiary: Time (Packing efficiency)
    scores = bottom_level + 0.1 * succ_count + 0.1 * times
    
    return scores
# EVOLVE-BLOCK-END
