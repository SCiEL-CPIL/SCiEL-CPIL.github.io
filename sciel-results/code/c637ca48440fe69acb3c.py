"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Compute a priority score for each activity in a resource-constrained project scheduling instance.
    
    The serial schedule-generation scheme starts the highest-priority eligible activity.
    We want to minimize makespan, so we should prioritize activities that are:
    1. On critical paths (longest path from start or to end)
    2. Have high resource demands (to avoid bottlenecks)
    3. Have few successors (to free up resources early)
    
    We use a combination of:
    - Longest path from start (forward pass)
    - Longest path to end (backward pass) 
    - Resource demand intensity
    - Successor count (fewer successors = higher priority)
    """
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Get predecessors and build successors
    predecessors = instance.get("predecessors", None)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
    
    # Get resource demands and capacities
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    if demands is None:
        demands = np.ones((n, 1))
    else:
        demands = np.asarray(demands, dtype=float)
        if demands.ndim == 1:
            demands = demands.reshape(-1, 1)
    
    if capacities is None:
        capacities = np.ones(demands.shape[1])
    else:
        capacities = np.asarray(capacities, dtype=float)
    
    # Forward pass: longest path from start (earliest start times)
    # ES[i] = max(ES[p] + durations[p]) for all predecessors p
    ES = np.zeros(n)
    # Topological order
    in_degree = np.array([len(preds) for preds in predecessors])
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    # Forward pass
    for i in topo_order:
        for pred in predecessors[i]:
            if ES[pred] + durations[pred] > ES[i]:
                ES[i] = ES[pred] + durations[pred]
    
    # Backward pass: longest path to end (latest finish times)
    # LF[i] = min(LS[s] - durations[s]) for all successors s
    # LS[i] = LF[i] - durations[i]
    # We want the longest path from i to the end
    LF = np.zeros(n)
    # Initialize with infinity
    LF[:] = np.inf
    
    # Process in reverse topological order
    for i in reversed(topo_order):
        if len(successors[i]) == 0:
            LF[i] = ES[i] + durations[i]
        else:
            min_LF = np.inf
            for succ in successors[i]:
                if LF[succ] - durations[succ] < min_LF:
                    min_LF = LF[succ] - durations[succ]
            LF[i] = min_LF
    
    # Total float: LF[i] - ES[i] - durations[i]
    # Activities with low total float are on critical paths
    total_float = LF - ES - durations
    # Normalize total float (lower is better, so we invert)
    max_tf = np.max(total_float) if np.max(total_float) > 0 else 1.0
    crit_score = 1.0 - (total_float / max_tf)  # Higher for critical activities
    
    # Resource demand intensity
    # For each activity, compute the sum of (demand / capacity) across all resources
    if demands.shape[1] > 0 and capacities.size > 0:
        resource_intensity = np.sum(demands / capacities, axis=1)
        max_intensity = np.max(resource_intensity) if np.max(resource_intensity) > 0 else 1.0
        resource_score = resource_intensity / max_intensity
    else:
        resource_score = np.ones(n)
    
    # Successor count: fewer successors means higher priority
    succ_counts = np.array([len(succs) for succs in successors])
    max_succ = np.max(succ_counts) if np.max(succ_counts) > 0 else 1.0
    succ_score = 1.0 - (succ_counts / max_succ)  # Higher for fewer successors
    
    # Duration score: longer durations get higher priority
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_score = durations / max_dur
    
    # Combine scores with weights
    # Critical path is most important, then duration, then resource demand
    weights = [0.4, 0.3, 0.2, 0.1]  # crit, dur, resource, succ
    priority_scores = (
        weights[0] * crit_score +
        weights[1] * dur_score +
        weights[2] * resource_score +
        weights[3] * succ_score
    )
    
    return priority_scores
# EVOLVE-BLOCK-END
