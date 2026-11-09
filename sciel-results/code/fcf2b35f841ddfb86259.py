# EVOLVE-BLOCK-START
import numpy as np


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
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors/successors
    # instance may have "predecessors" as a list of lists or similar
    preds = instance.get("predecessors", None)
    
    # Build successor map
    succs = [[] for _ in range(n)]
    if preds is not None:
        for j, pred_list in enumerate(preds):
            for i in pred_list:
                succs[i].append(j)
    
    # Compute critical path length (longest path to end) for each activity
    # Using topological order (reverse topological for longest path to sink)
    # in_degree
    in_deg = [0] * n
    if preds is not None:
        for j, pred_list in enumerate(preds):
            in_deg[j] = len(pred_list)
    
    # Topological sort (Kahn's algorithm)
    from collections import deque
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succs[node]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    
    # If topological sort didn't include all nodes (cycle), use all nodes
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Longest path from each node to any sink (reverse topological)
    cp_len = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        if not succs[node]:
            cp_len[node] = durations[node]
        else:
            cp_len[node] = durations[node] + max(cp_len[s] for s in succs[node])
    
    # Resource demand intensity
    # instance may have "demands" as a list of lists (per activity, per resource)
    # and "capacities" as a list (per resource)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    demand_score = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        caps_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 1:
            # Single resource or scalar
            if len(caps_arr) == 1 or caps_arr.size == 1:
                cap = caps_arr.flat[0] if caps_arr.size > 0 else 1.0
                if cap > 0:
                    demand_score = demands_arr / cap
                else:
                    demand_score = demands_arr
            else:
                # Mismatch, just use raw demands
                demand_score = demands_arr
        elif demands_arr.ndim == 2:
            # demands_arr[i, k] for activity i, resource k
            # Normalize by capacity per resource
            if len(caps_arr) == demands_arr.shape[1]:
                norm = demands_arr / np.maximum(caps_arr, 1e-9)
                # Use max normalized demand per activity
                demand_score = np.max(norm, axis=1)
            else:
                demand_score = np.max(demands_arr, axis=1)
    
    # Successor count (more successors = more critical)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # Normalize each component to [0, 1] range
    def normalize(x):
        x = np.asarray(x, dtype=float)
        if x.max() - x.min() < 1e-9:
            return np.ones_like(x) * 0.5
        return (x - x.min()) / (x.max() - x.min())
    
    cp_norm = normalize(cp_len)
    dur_norm = normalize(durations)
    dem_norm = normalize(demand_score)
    succ_norm = normalize(succ_count)
    
    # Weighted combination
    # Critical path is most important, then duration, then demand, then successor count
    score = (
        0.45 * cp_norm +
        0.25 * dur_norm +
        0.20 * dem_norm +
        0.10 * succ_norm
    )
    
    return score
# EVOLVE-BLOCK-END
