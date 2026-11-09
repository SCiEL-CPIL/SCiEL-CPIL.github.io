"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using Ranked Positional Weight (RPW) and Task Time.
    
    Approach:
    1. Calculate RPW for each task: sum of its own time and the times of all its successors.
       This differs from Longest Path (critical path) by summing all branches,
       favoring tasks that unlock a large amount of subsequent work.
    2. Combine RPW with Task Time to ensure long tasks are prioritized when RPW is similar.
    3. Add a small bonus for tasks with many direct successors to break ties.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor lists for efficient traversal
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)
                
    # Compute RPW (Ranked Positional Weight)
    # RPW[i] = times[i] + sum(RPW[j] for j in successors[i])
    # Process in reverse topological order
    rpw = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        rpw[node] = times[node]
        for succ in successors[node]:
            rpw[node] += rpw[succ]
            
    # Compute direct successor counts
    succ_counts = np.array([len(succs) for succs in successors], dtype=float)
    
    # Normalize components
    # RPW is the primary driver
    rpw_norm = rpw / (rpw.max() + 1e-9)
    time_norm = times / (times.max() + 1e-9)
    succ_norm = succ_counts / (succ_counts.max() + 1e-9)
    
    # Weighted combination
    # RPW captures the "downstream load" comprehensively.
    # Time ensures that if two tasks have similar downstream loads, the longer one is picked.
    # SuccCount helps break ties by preferring tasks that unlock more immediate options.
    score = 0.8 * rpw_norm + 0.15 * time_norm + 0.05 * succ_norm
    
    return score
# EVOLVE-BLOCK-END
