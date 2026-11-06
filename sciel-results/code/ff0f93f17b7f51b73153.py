"""Rewritten program for Resource-Constrained Project Scheduling (serial SGS).

Strategy: Use a composite priority that combines:
1. CPM total float (critical path slack) - lower float = higher priority
2. Resource scarcity pressure - activities using scarce resources get higher priority
3. Most Preceding Successors (MPS) - activities with more successors get higher priority
4. Duration as a tiebreaker

This approach prioritizes activities that are on the critical path and that consume
scarce resources, which typically shortens the makespan in serial SGS.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cpm_floats(durations, pred_lists, n):
    """Compute earliest start (ES), latest start (LS), and total float for each activity.
    
    Returns:
        es: array of earliest start times
        ls: array of latest start times
        float_arr: array of total floats (ls - es)
    """
    # Build successor lists
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_lists[i]:
            succ_lists[p].append(i)
    
    # Forward pass: compute ES
    es = np.zeros(n, dtype=float)
    # Topological order via Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_lists[i]:
            in_degree[i] += 1
    
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succ_lists[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If there's a cycle, fall back to simple order
    if len(topo_order) < n:
        topo_order = list(range(n))
    
    # Forward pass
    for u in topo_order:
        for v in succ_lists[u]:
            es[v] = max(es[v], es[u] + durations[u])
    
    # Project makespan (unconstrained)
    makespan = max(es[i] + durations[i] for i in range(n)) if n > 0 else 0.0
    
    # Backward pass: compute LS
    ls = np.full(n, float(makespan), dtype=float)
    for u in reversed(topo_order):
        for v in succ_lists[u]:
            ls[u] = min(ls[u], ls[v] - durations[u])
    
    float_arr = ls - es
    return es, ls, float_arr, makespan


def priority(instance):
    """Compute priority scores for each activity. Higher score = higher priority."""
    start_time = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Parse predecessors
    preds = instance.get("predecessors", None)
    if preds is None:
        pred_lists = [[] for _ in range(n)]
    else:
        pred_lists = []
        for i in range(n):
            p = preds[i]
            if p is None:
                p = []
            if isinstance(p, (list, tuple, np.ndarray)):
                pred_lists.append([int(x) for x in p])
            elif isinstance(p, (int, np.integer)):
                pred_lists.append([int(p)])
            else:
                pred_lists.append([])
    
    # Parse resource demands and capacities
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # Compute CPM floats
    es, ls, float_arr, makespan_cpm = _compute_cpm_floats(durations, pred_lists, n)
    
    # Normalize float to [0, 1] range (lower float = higher priority)
    max_float = np.max(float_arr) if n > 0 else 1.0
    if max_float > 0:
        normalized_float = 1.0 - float_arr / max_float
    else:
        normalized_float = np.ones(n, dtype=float)
    
    # Resource scarcity: for each activity, compute the ratio of demand to capacity
    # for each resource it uses. Higher ratio = more critical resource usage.
    resource_pressure = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        # demands_arr shape: (n, num_resources) or (num_resources,) if single resource
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(n, 1)
        if capacities_arr.ndim == 1:
            capacities_arr = capacities_arr.reshape(1, -1)
        
        # For each activity, compute max(demand_j / capacity_j) over resources it uses
        for i in range(n):
            d = demands_arr[i]
            c = capacities_arr[0] if capacities_arr.shape[0] == 1 else capacities_arr[i]
            # Avoid division by zero
            valid = c > 0
            if np.any(valid):
                ratios = np.where(valid, d / np.where(valid, c, 1.0), 0.0)
                resource_pressure[i] = np.max(ratios)
    
    # Normalize resource pressure
    max_rp = np.max(resource_pressure) if n > 0 else 1.0
    if max_rp > 0:
        normalized_rp = resource_pressure / max_rp
    else:
        normalized_rp = np.zeros(n, dtype=float)
    
    # Most Preceding Successors (MPS): count successors
    succ_counts = np.zeros(n, dtype=float)
    for i in range(n):
        for p in pred_lists[i]:
            succ_counts[p] += 1
    
    max_succ = np.max(succ_counts) if n > 0 else 1.0
    if max_succ > 0:
        normalized_succ = succ_counts / max_succ
    else:
        normalized_succ = np.zeros(n, dtype=float)
    
    # Duration normalization
    max_dur = np.max(durations) if n > 0 else 1.0
    if max_dur > 0:
        normalized_dur = durations / max_dur
    else:
        normalized_dur = np.zeros(n, dtype=float)
    
    # Composite priority: weighted sum
    # Higher weight on critical path (low float) and resource scarcity
    w_float = 0.45
    w_rp = 0.30
    w_succ = 0.15
    w_dur = 0.10
    
    scores = (w_float * normalized_float + 
              w_rp * normalized_rp + 
              w_succ * normalized_succ + 
              w_dur * normalized_dur)
    
    # Safety: ensure we're well within time limit
    elapsed = time.time() - start_time
    if elapsed > 100:  # Well before 120s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
