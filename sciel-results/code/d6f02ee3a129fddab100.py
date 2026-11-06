# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Strategy: Combine multiple heuristics that are known to work well for RCPSP:
    1. Most Immediate Successors (MIS): activities with many successors should go early
    2. Total Float (slack): activities with less slack (more critical) should go early
    3. Resource demand intensity: activities using scarce resources should go early
    4. Duration: longer activities should go early (to avoid blocking)
    
    We compute a composite score that balances these factors.
    """
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors
    preds = instance.get("predecessors", None)
    if preds is None:
        preds = [[] for _ in range(n)]
    
    # Normalize predecessors to lists of ints
    pred_lists = []
    for i in range(n):
        p = preds[i]
        if p is None:
            p = []
        pred_lists.append(list(p))
    
    # Build successor lists
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                succ_lists[p].append(i)
    
    # Compute earliest start times (ES) via forward pass
    ES = np.zeros(n, dtype=float)
    in_degree = np.array([len(pred_lists[i]) for i in range(n)])
    completed = [False] * n
    
    # Topological order using Kahn's algorithm
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # Pick any node (for determinism, pick smallest index)
        queue.sort()
        u = queue.pop(0)
        topo_order.append(u)
        completed[u] = True
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If not all nodes were processed, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use duration-based priority
        return np.asarray(durations, dtype=float)
    
    # Forward pass for ES
    for u in topo_order:
        for v in succ_lists[u]:
            ES[v] = max(ES[v], ES[u] + durations[u])
    
    # Compute latest finish times (LF) via backward pass
    # Project end time = max ES[i] + durations[i]
    project_end = max(ES[i] + durations[i] for i in range(n)) if n > 0 else 0.0
    LF = np.full(n, project_end, dtype=float)
    
    # Backward pass
    for u in reversed(topo_order):
        for v in succ_lists[u]:
            LF[u] = min(LF[u], LF[v] - durations[u])
    
    # Total float = LF - ES - duration
    TF = LF - ES - durations
    
    # Most Immediate Successors count
    MIS = np.array([len(succ_lists[i]) for i in range(n)], dtype=float)
    
    # Number of immediate predecessors
    NIP = np.array([len(pred_lists[i]) for i in range(n)], dtype=float)
    
    # Resource demand intensity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # If demands is 2D (n x num_resources) and capacities is 1D (num_resources)
        if demands_arr.ndim == 2:
            # For each activity, compute the maximum ratio of demand to capacity across resources
            # This measures how "scarce" the resources are for this activity
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = demands_arr / capacities_arr[np.newaxis, :]
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=1.0)
            resource_intensity = np.max(ratio, axis=1)
        else:
            # Single resource case
            with np.errstate(divide='ignore', invalid='ignore'):
                resource_intensity = demands_arr / capacities_arr
            resource_intensity = np.nan_to_num(resource_intensity, nan=0.0, posinf=1.0)
    else:
        resource_intensity = np.zeros(n, dtype=float)
    
    # Normalize each component to [0, 1] range
    def normalize(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min < 1e-10:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    MIS_norm = normalize(MIS)
    TF_norm = normalize(TF)
    dur_norm = normalize(durations)
    ri_norm = normalize(resource_intensity)
    NIP_norm = normalize(NIP)
    
    # Invert total float: less float = higher priority
    TF_inv = 1.0 - TF_norm
    
    # Composite score
    # Weights based on literature: critical path (TF) and MIS are most important
    score = (
        0.35 * TF_inv +      # Critical path activities first
        0.25 * MIS_norm +    # Many successors first
        0.15 * dur_norm +    # Long duration first
        0.15 * ri_norm +     # High resource intensity first
        0.10 * NIP_norm      # Many predecessors first (already "ready" in some sense)
    )
    
    # Ensure higher score = higher priority
    return np.asarray(score, dtype=float)

# EVOLVE-BLOCK-END
