"""Improved program for RCPSP using combined heuristic."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque


def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    # Extract instance components with defensive key handling
    # Try common key names for predecessors
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        # Fallback: no precedence constraints
        return np.asarray(durations, dtype=float)
    
    # Build successor lists
    succs = [set() for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            succs[i].add(j)
    
    # Resource data
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # Forward pass: earliest start times
    in_degree = [len(preds[j]) for j in range(n)]
    es = np.zeros(n, dtype=float)
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            es[v] = max(es[v], es[u] + durations[u])
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # Backward pass: latest start times
    # Compute project duration
    makespan = max(es) if n > 0 else 0.0
    ls = np.full(n, makespan, dtype=float)
    # Process in reverse topological order
    reverse_topo = list(reversed(topo_order))
    for u in reverse_topo:
        for v in succs[u]:
            ls[u] = min(ls[u], ls[v] - durations[u])
    
    # Total float
    total_float = ls - es
    
    # Resource pressure: sum of demand/capacity ratios
    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands, dtype=float)
        capacities = np.asarray(capacities, dtype=float)
        if demands.ndim == 2:
            # demands shape: (n_activities, n_resources)
            safe_cap = np.where(capacities > 0, capacities, 1.0)
            ratios = demands / safe_cap
            resource_pressure = ratios.sum(axis=1)
        else:
            resource_pressure = np.zeros(n, dtype=float)
    
    # Successor count (immediate)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    
    # Combine heuristics
    # Higher priority = lower total float, higher resource pressure, more successors
    # Normalize each component
    tf_max = total_float.max() if total_float.max() > 0 else 1.0
    tf_score = 1.0 / (total_float + 1.0)  # Inverse float: lower float -> higher score
    
    rp_max = resource_pressure.max() if resource_pressure.max() > 0 else 1.0
    rp_score = resource_pressure / rp_max
    
    sc_max = succ_count.max() if succ_count.max() > 0 else 1.0
    sc_score = succ_count / sc_max
    
    # Weighted combination
    priority_scores = 0.5 * tf_score + 0.3 * rp_score + 0.2 * sc_score
    
    return priority_scores
# EVOLVE-BLOCK-END
