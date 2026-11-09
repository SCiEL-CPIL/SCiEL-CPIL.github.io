"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    """
    Score each assembly task using a static heuristic approximating tightest-fit packing.
    
    We use a weighted combination of:
    1. Processing Time (LPT): Larger tasks are prioritized to fill stations efficiently.
    2. Ranked Positional Weight (RPW): Tasks with more "downstream" work are prioritized.
    3. Immediate Successor Count (ISC): Tasks that unlock many others are prioritized.
    
    This approximates the 'negative slack' directive by prioritizing tasks that 
    significantly contribute to filling the cycle time and are critical to the flow.
    """
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Build successor list and in-degrees
    succs = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            if 0 <= p < n:
                succs[p].append(i)
                in_degree[i] += 1
                
    # Topological sort
    q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            q.append(i)
            
    topo_order = []
    in_degree_temp = in_degree[:]
    while q:
        node = q.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_degree_temp[s] -= 1
            if in_degree_temp[s] == 0:
                q.append(s)
                
    if len(topo_order) != n:
        return times.copy()
        
    # Compute RPW
    rpW = np.zeros(n, dtype=float)
    for i in reversed(topo_order):
        rpW[i] = times[i]
        for s in succs[i]:
            rpW[i] += rpW[s]
            
    # Compute Immediate Successor Count
    isc = np.array([len(s) for s in succs], dtype=float)
    
    # Normalize components
    def normalize(arr):
        min_val, max_val = arr.min(), arr.max()
        range_val = max_val - min_val
        if range_val < 1e-10:
            return np.ones_like(arr)
        return (arr - min_val) / range_val
        
    time_norm = normalize(times)
    rpW_norm = normalize(rpW)
    isc_norm = normalize(isc)
    
    # Weighted combination
    # LPT is strong for packing. RPW is strong for ALB structure.
    # ISC helps with "fan-out" criticality.
    priority_scores = 0.4 * time_norm + 0.4 * rpW_norm + 0.2 * isc_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
