# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Uses a combination of:
    1. Critical path length (longest path from activity to end)
    2. Resource demand intensity (higher demand = harder to schedule, prioritize)
    3. Duration (longer tasks take more time)
    4. Successor count (activities with many successors are more critical)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors/successors
    preds = instance.get("predecessors", None)
    
    # Build successor list
    succs = [[] for _ in range(n)]
    if preds is not None:
        for i in range(n):
            if i < len(preds):
                for p in preds[i]:
                    if p is not None and 0 <= p < n:
                        succs[p].append(i)
    
    # Compute longest path from each activity to end (critical path length)
    # Using topological order via Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        if i < len(preds):
            for p in preds[i]:
                if p is not None and 0 <= p < n:
                    in_degree[i] += 1
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_degree_copy = in_degree.copy()
    
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_degree_copy[s] -= 1
            if in_degree_copy[s] == 0:
                queue.append(s)
    
    # If not all nodes are in topo_order, add remaining (handle cycles gracefully)
    if len(topo_order) < n:
        for i in range(n):
            if i not in set(topo_order):
                topo_order.append(i)
    
    # Compute longest path from each node to end (reverse topological order)
    longest_to_end = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if succs[node]:
            longest_to_end[node] = durations[node] + np.max([longest_to_end[s] for s in succs[node]])
        else:
            longest_to_end[node] = durations[node]
    
    # Compute longest path from start to each node
    longest_from_start = np.zeros(n, dtype=float)
    for node in topo_order:
        if preds is not None and node < len(preds) and preds[node]:
            valid_preds = [p for p in preds[node] if p is not None and 0 <= p < n]
            if valid_preds:
                longest_from_start[node] = np.max([longest_from_start[p] + durations[p] for p in valid_preds])
        # else remains 0 (or we could set to durations[node] but it's a start node)
    
    # Resource demands
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        
        # Resource demand intensity: sum of (demand/capacity) for each resource
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            intensity = np.sum(np.where(capacities > 0, demands / capacities, 0), axis=1)
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        intensity = np.zeros(n, dtype=float)
    
    # Successor count
    succ_count = np.array([len(s) for s in succs], dtype=float)
    
    # Combine scores with weights
    # Normalize each component
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_lte = normalize(longest_to_end)
    norm_lfs = normalize(longest_from_start)
    norm_dur = normalize(durations)
    norm_int = normalize(intensity)
    norm_sc = normalize(succ_count)
    
    # Weighted combination
    # Critical path length is most important
    w_lte = 0.35
    w_lfs = 0.15
    w_dur = 0.20
    w_int = 0.15
    w_sc = 0.15
    
    scores = (w_lte * norm_lte + 
              w_lfs * norm_lfs + 
              w_dur * norm_dur + 
              w_int * norm_int + 
              w_sc * norm_sc)
    
    return scores
# EVOLVE-BLOCK-END
