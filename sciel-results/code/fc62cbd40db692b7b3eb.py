"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import heapq


def priority(instance):
    """
    Score each activity based on a resource-bottleneck heuristic.
    
    The idea: activities that consume scarce resources (high demand relative to capacity)
    and have long durations should be prioritized, especially if they are on critical paths.
    
    We use a composite heuristic:
    1. Latest start time (LST) from backward pass - activities with smaller LST are more urgent
    2. Resource pressure: sum of demand/capacity ratios for each resource
    3. Duration weighted by resource scarcity
    
    This combines critical path logic with resource contention awareness.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)  # shape (n, R)
    capacities = np.asarray(instance["capacities"], dtype=float)  # shape (R,)
    predecessors = instance["predecessors"]  # list of lists, preds[i] = list of predecessor indices
    
    n = len(durations)
    R = len(capacities)
    
    # Build successor lists
    successors = defaultdict(list)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        for p in preds:
            successors[p].append(i)
    
    # Forward pass: earliest start and finish
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
    
    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
    
    # If topological sort didn't include all nodes (cycle), handle gracefully
    if len(topo_order) < n:
        # Fallback: just use duration-based priority
        return durations.copy()
    
    # Forward pass
    for u in topo_order:
        for v in successors[u]:
            if ef[u] > es[v]:
                es[v] = ef[u]
        ef[u] = es[u] + durations[u]
    
    # Backward pass: latest start and finish
    project_end = max(ef) if n > 0 else 0
    ls = np.full(n, project_end, dtype=float)
    lf = np.full(n, project_end, dtype=float)
    out_degree = np.zeros(n, dtype=int)
    for i in range(n):
        out_degree[i] = len(successors[i])
    
    # Reverse topological order
    reverse_topo = list(reversed(topo_order))
    temp_out_degree = out_degree.copy()
    for u in reverse_topo:
        for v in successors[u]:
            if ls[v] < lf[u]:
                lf[u] = ls[v]
        ls[u] = lf[u] - durations[u]
    
    # Slack (total float)
    slack = ls - es
    
    # Resource pressure: for each activity, how much of each resource it consumes
    # relative to capacity. Higher means more bottleneck.
    if R > 0 and np.all(capacities > 0):
        # demand_ratio[i, r] = demands[i, r] / capacities[r]
        demand_ratio = demands / capacities[np.newaxis, :]
        # Sum across resources, weighted by duration
        resource_pressure = np.sum(demand_ratio * durations[:, np.newaxis], axis=1)
    else:
        resource_pressure = np.zeros(n, dtype=float)
    
    # Criticality: inverse slack (smaller slack = more critical)
    # Add a small epsilon to avoid division by zero
    max_slack = np.max(slack) if n > 0 else 1.0
    if max_slack > 0:
        criticality = 1.0 / (slack + 1.0)
    else:
        criticality = np.ones(n, dtype=float)
    
    # Combine: prioritize activities that are critical (low slack) AND have high resource pressure
    # Also weight by duration since longer activities benefit more from early scheduling
    base_score = durations * criticality
    resource_score = resource_pressure * criticality
    
    # Normalize components
    if np.max(base_score) > 0:
        base_score_norm = base_score / np.max(base_score)
    else:
        base_score_norm = np.zeros(n, dtype=float)
    
    if np.max(resource_score) > 0:
        resource_score_norm = resource_score / np.max(resource_score)
    else:
        resource_score_norm = np.zeros(n, dtype=float)
    
    # Final priority: blend of critical duration and resource pressure
    # Give more weight to resource pressure since this is a resource-constrained problem
    priority_scores = 0.4 * base_score_norm + 0.6 * resource_score_norm
    
    # Add a small tie-breaker based on duration to ensure consistent ordering
    if np.max(durations) > 0:
        priority_scores += 0.1 * (durations / np.max(durations))
    
    return priority_scores
# EVOLVE-BLOCK-END
