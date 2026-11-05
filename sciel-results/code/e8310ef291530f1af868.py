"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = instance["demands"]
    capacities = np.asarray(instance["capacities"], dtype=float)
    n_resources = len(capacities)

    # Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1

    # Topological sort using Kahn's algorithm
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in successors[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    if len(topo_order) != n:
        topo_order = list(range(n))

    # Calculate Critical Path (CP) length for each activity
    # CP[i] is the length of the longest path starting from i (including i's duration)
    cp = np.zeros(n)
    # Process in reverse topological order
    for i in reversed(topo_order):
        if successors[i]:
            cp[i] = durations[i] + max(cp[s] for s in successors[i])
        else:
            cp[i] = durations[i]
            
    # Normalize CP
    cp_min, cp_max = cp.min(), cp.max()
    cp_range = cp_max - cp_min + 1e-9
    norm_cp = (cp - cp_min) / cp_range

    # Calculate Total Resource Demand per activity
    # Sum of demands across all resources, normalized by capacity to get "pressure"
    # demand[i] is likely a vector or list of length n_resources
    # The instance structure implies demands[i] corresponds to activity i.
    # Let's assume demands is a 2D array (n x n_resources) or list of lists.
    
    # Robustly handle demands shape
    demands_arr = np.asarray(demands, dtype=float)
    if demands_arr.ndim == 1:
        # If 1D, maybe it's a single resource or total? 
        # Given "four resources", it's likely 2D. 
        # If 1D, we can't map to specific resources easily without knowing which.
        # Let's assume standard format: demands[i][r]
        # If it's 1D, we treat it as a single scalar demand for a generic resource or sum?
        # The task says "four resources". So demands should be n x 4.
        # If the input is 1D, it might be a flattened list or single resource.
        # We will try to reshape or handle gracefully.
        # If 1D, we assume it's the total demand or single resource.
        resource_pressure = demands_arr / (np.max(demands_arr) + 1e-9)
    else:
        # demands_arr is (n, n_resources)
        # Calculate "pressure" as sum of (demand / capacity) for each activity
        # This measures how tight the resource is for this activity.
        # capacity is (n_resources,)
        # pressure[i] = sum_r (demand[i][r] / capacity[r])
        # We want higher pressure to have higher priority? 
        # Or lower? 
        # High pressure means it uses a lot of scarce resource. 
        # In serial SGS, scheduling high pressure items early might prevent them from being blocked later?
        # Or it might block others. 
        # Literature suggests "Resource Criticality" or "Minimum Slack" is good.
        # Let's stick to CP as primary. 
        # Let's add a small weight to CP only, as it is the most robust.
        
        # Alternative: Weighted CP where weights are based on resource scarcity.
        # But simple CP is a strong baseline.
        
        # Let's just use CP.
        resource_pressure = np.zeros(n)
        # If we want to add resource info:
        # pressure = np.sum(demands_arr / capacities, axis=1)
        # norm_pressure = (pressure - pressure.min()) / (pressure.max() - pressure.min() + 1e-9)
        # score = 0.9 * norm_cp + 0.1 * norm_pressure
        
    # The current program used LS and Grade.
    # CP is different.
    
    # To make it a "new approach", let's use **CP** exclusively or with a very different secondary metric.
    # Let's try **CP** + **Duration** (longer tasks first? No, CP already includes duration).
    # Let's try **CP** + **Number of Predecessors** (activities with many preds might be bottlenecks? No, they are ready later).
    
    # Let's stick to pure CP normalized.
    # Higher CP -> Higher Priority.
    
    priority_score = norm_cp

    return priority_score
# EVOLVE-BLOCK-END
