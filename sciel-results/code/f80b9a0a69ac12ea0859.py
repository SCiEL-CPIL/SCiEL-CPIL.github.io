"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Score each activity using a Resource-Scarcity and Graph-Breadth heuristic.
    
    This approach differs from Critical Path (LRP) methods by focusing on:
    1. Weighted Resource Demand: Activities consuming scarce resources (high demand/capacity ratio) are prioritized to prevent resource starvation later.
    2. Total Successors: Activities that unlock many downstream tasks are prioritized to maintain schedule flow and reduce idle time.
    """
    n = len(instance["durations"])
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = instance["demands"]
    capacities = instance["capacities"]
    
    # 1. Compute Weighted Resource Demand (WRD)
    # demands: (n, R), capacities: (R,)
    demands_arr = np.asarray(demands, dtype=float)
    capacities_arr = np.asarray(capacities, dtype=float)
    
    # Avoid division by zero
    safe_caps = np.where(capacities_arr == 0, 1.0, capacities_arr)
    # WRD[i] = sum_r (d_{ir} / C_r)
    wrd = np.sum(demands_arr / safe_caps, axis=1)
    
    # 2. Compute Total Successors (TS)
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            if i is not None:
                successors[i].append(j)
    
    # Count reachable nodes for each activity
    ts = np.zeros(n, dtype=int)
    for i in range(n):
        # BFS/DFS to count all successors
        visited = set()
        queue = deque(successors[i])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(successors[node])
        ts[i] = len(visited)
    
    # 3. Normalize scores
    max_wrd = np.max(wrd) if np.max(wrd) > 0 else 1.0
    max_ts = np.max(ts) if np.max(ts) > 0 else 1.0
    
    norm_wrd = wrd / max_wrd
    norm_ts = ts.astype(float) / max_ts
    
    # 4. Combine
    # Weighting: Resource scarcity is often the primary bottleneck in RCPSP.
    # Breadth helps with flow.
    score = 0.6 * norm_wrd + 0.4 * norm_ts
    
    return score
# EVOLVE-BLOCK-END
