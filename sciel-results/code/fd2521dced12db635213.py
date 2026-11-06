import numpy as np
import time

def priority(instance):
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    
    # 1. Build successor lists
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
            
    # 2. Topological Sort (Kahn's Algorithm)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    current_queue = queue[:]
    in_degree_temp = in_degree[:]
    
    while current_queue:
        # Pick a node (order doesn't matter for correctness of L_start/L_end calculation, 
        # but for determinism, we can sort or just pop)
        node = current_queue.pop(0) 
        topo_order.append(node)
        for succ in successors[node]:
            in_degree_temp[succ] -= 1
            if in_degree_temp[succ] == 0:
                current_queue.append(succ)
                
    if len(topo_order) != n:
        # Cycle detected? Fallback to durations
        return durations.copy()

    # 3. Compute L_start (Longest path from start to i)
    L_start = np.zeros(n)
    for i in topo_order:
        # L_start[i] is duration[i] + max(L_start[p]) for p in predecessors[i]
        max_pred = 0.0
        for p in predecessors[i]:
            if L_start[p] > max_pred:
                max_pred = L_start[p]
        L_start[i] = durations[i] + max_pred
        
    # 4. Compute L_end (Longest path from i to end)
    L_end = np.zeros(n)
    # Iterate in reverse topological order
    for i in reversed(topo_order):
        # L_end[i] is duration[i] + max(L_end[s]) for s in successors[i]
        max_succ = 0.0
        for s in successors[i]:
            if L_end[s] > max_succ:
                max_succ = L_end[s]
        L_end[i] = durations[i] + max_succ
        
    # 5. Calculate Priority Score
    # Criticality = Path from start to end through i
    criticality = L_start + L_end - durations
    
    # Resource pressure: prioritize activities with higher resource demand relative to capacity
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    # Normalize resource usage: sum of (demand/capacity) for each activity
    # This captures how much each activity stresses the resource pool
    resource_pressure = np.sum(demands / capacities[None, :], axis=1)
    
    # Normalize both criticality and resource pressure to [0, 1] for balanced weighting
    crit_min, crit_max = np.min(criticality), np.max(criticality)
    rp_min, rp_max = np.min(resource_pressure), np.max(resource_pressure)
    
    if crit_max > crit_min:
        crit_norm = (criticality - crit_min) / (crit_max - crit_min)
    else:
        crit_norm = np.ones_like(criticality)
    
    if rp_max > rp_min:
        rp_norm = (resource_pressure - rp_min) / (rp_max - rp_min)
    else:
        rp_norm = np.zeros_like(resource_pressure)
    
    # Combine: weight criticality more heavily, but resource pressure can break ties
    # and push bottleneck activities earlier
    alpha = 0.85  # weight for criticality
    beta = 0.15   # weight for resource pressure
    
    scores = alpha * crit_norm + beta * rp_norm
    
    # Add small duration component to break remaining ties (longer tasks first)
    dur_min, dur_max = np.min(durations), np.max(durations)
    if dur_max > dur_min:
        dur_norm = (durations - dur_min) / (dur_max - dur_min)
    else:
        dur_norm = np.zeros_like(durations)
    
    scores = scores + 0.05 * dur_norm
    
    return scores
