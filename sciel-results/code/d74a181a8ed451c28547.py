"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute priority scores using Critical Path (CP) and Positional Weight (PW).
    CP focuses on the longest dependency chain, while PW focuses on total work.
    We combine them to balance chain length and total volume.
    """
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Parse predecessors
    pred = instance["predecessors"]
    
    # Build successor lists
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in pred[i]:
            succ[p].append(i)
    
    # Compute in-degrees for topological sort
    in_degree = [0] * n
    for i in range(n):
        for p in pred[i]:
            in_degree[i] += 1
            
    # Topological sort (Kahn's algorithm)
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    in_degree_copy = in_degree[:]
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree_copy[s] -= 1
            if in_degree_copy[s] == 0:
                q.append(s)
                
    if len(topo_order) != n:
        return times.copy()
    
    # Calculate Positional Weights (PW) and Critical Path (CP) in reverse topological order
    pos_weight = np.zeros(n, dtype=float)
    crit_path = np.zeros(n, dtype=float)
    
    for node in reversed(topo_order):
        total_pw = 0.0
        max_cp = 0.0
        for s in succ[node]:
            total_pw += pos_weight[s]
            if crit_path[s] > max_cp:
                max_cp = crit_path[s]
        
        pos_weight[node] = times[node] + total_pw
        crit_path[node] = times[node] + max_cp
        
    # Normalize
    max_pw = pos_weight.max() if pos_weight.max() > 0 else 1.0
    max_cp = crit_path.max() if crit_path.max() > 0 else 1.0
    
    pw_norm = pos_weight / max_pw
    cp_norm = crit_path / max_cp
    
    # Combine: Equal weight for now, or favor CP?
    # CP ensures we don't get stuck with a long chain that doesn't fit.
    # PW ensures we fill capacity with total work.
    # Let's try 0.5 PW + 0.5 CP
    priority_scores = 0.5 * pw_norm + 0.5 * cp_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
