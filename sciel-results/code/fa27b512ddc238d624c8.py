"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    
    Strategy: Use a composite priority that balances:
    1. Total remaining work (sum of durations of all downstream activities including self)
    2. Resource pressure (how much of the scarce resource this activity uses)
    3. Precedence depth (how early in the project this activity sits)
    
    The priority is designed so that the serial SGS picks activities that:
    - Complete the most remaining project work
    - Don't over-constrain resources for subsequent activities
    - Are on the critical path
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors and successors
    # instance["predecessors"] is typically a list of lists: predecessors[i] = list of j such that j must finish before i starts
    predecessors = instance.get("predecessors", [list() for _ in range(n)])
    
    # Build successor lists
    successors = [[] for _ in range(n)]
    for i in range(n):
        for pred in predecessors[i]:
            successors[pred].append(i)
    
    # Compute total remaining work: for each activity i, sum of durations of i and all its descendants
    # Use topological order (reverse topological for post-order traversal)
    # First, compute in-degrees and topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for pred in predecessors[i]:
            in_degree[i] += 1
    
    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree_copy[v] -= 1
            if in_degree_copy[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't have all nodes, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Add remaining nodes (shouldn't happen in valid instances)
        seen = set(topo_order)
        for i in range(n):
            if i not in seen:
                topo_order.append(i)
    
    # Compute total remaining work in reverse topological order
    total_remaining = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        total_remaining[u] = durations[u] + sum(total_remaining[v] for v in successors[u])
    
    # Compute resource pressure: for each activity, max ratio of demand to capacity across resources
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # Ensure shapes are compatible
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(n, 1)
        if capacities_arr.ndim == 1:
            capacities_arr = capacities_arr.reshape(-1, 1)
        
        # Resource pressure: max over resources of (demand / capacity)
        # This measures how "expensive" each activity is in terms of resource consumption
        with np.errstate(divide='ignore', invalid='ignore'):
            pressure = np.max(demands_arr / capacities_arr, axis=1)
        pressure = np.nan_to_num(pressure, nan=0.0, posinf=1.0)
    else:
        pressure = np.zeros(n, dtype=float)
    
    # Compute precedence depth: number of levels from start
    # Activities with higher depth are "later" in the project
    depth = np.zeros(n, dtype=float)
    for u in topo_order:
        if predecessors[u]:
            depth[u] = 1 + max(depth[p] for p in predecessors[u])
        else:
            depth[u] = 0
    
    max_depth = max(depth) if max(depth) > 0 else 1.0
    
    # Composite priority:
    # We want to prioritize activities that:
    # - Have high total remaining work (they unlock a lot of downstream work)
    # - Have low resource pressure (they don't consume too much scarce resource)
    # - Are on earlier levels (they can start sooner and keep the pipeline moving)
    
    # Normalize each component to [0, 1]
    tr_min, tr_max = total_remaining.min(), total_remaining.max()
    if tr_max > tr_min:
        tr_norm = (total_remaining - tr_min) / (tr_max - tr_min)
    else:
        tr_norm = np.ones(n, dtype=float)
    
    p_min, p_max = pressure.min(), pressure.max()
    if p_max > p_min:
        p_norm = (pressure - p_min) / (p_max - p_min)
    else:
        p_norm = np.zeros(n, dtype=float)
    
    d_norm = depth / max_depth
    
    # Weighted combination:
    # High total remaining work is good (positive)
    # Low pressure is good (negative weight on pressure)
    # Low depth is slightly good (negative weight on depth, to prefer early activities)
    priority_scores = (
        2.0 * tr_norm - 1.0 * p_norm - 0.3 * d_norm
    )
    
    # Add a small tie-breaker based on duration to avoid ties
    dur_min, dur_max = durations.min(), durations.max()
    if dur_max > dur_min:
        dur_norm = (durations - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.zeros(n, dtype=float)
    priority_scores += 0.1 * dur_norm
    
    return priority_scores
# EVOLVE-BLOCK-END
