import numpy as np
from collections import defaultdict, deque
import time

_START_TIME = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit


def priority(instance):
    t0 = time.time()
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Get predecessors with defensive key handling
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break
    
    if preds is None:
        preds = [[] for _ in range(n)]
    
    # Normalize preds to list of lists of ints
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()
    
    # Build successor list
    succs = [[] for _ in range(n)]
    for j in range(n):
        if j < len(preds):
            p_list = preds[j]
            if p_list is None:
                continue
            for i in p_list:
                if i is not None:
                    i = int(i)
                    if 0 <= i < n and i != j:
                        if j not in succs[i]:
                            succs[i].append(j)
    
    # Topological sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    # If not all nodes in topo_order, there's a cycle; add remaining
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
    
    # Forward pass: ES
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]
    
    project_duration = np.max(ES + durations) if n > 0 else 0
    
    # Backward pass: LS
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]
    
    total_float = LS - ES
    
    # Resource scarcity
    demands = None
    capacities = None
    for key in ["demands", "resource_demands", "demand", "resource_demand"]:
        if key in instance:
            demands = instance[key]
            break
    for key in ["capacities", "resource_capacities", "capacity", "resource_capacity"]:
        if key in instance:
            capacities = instance[key]
            break
    
    scarcity = np.zeros(n, dtype=float)
    num_resources = 0
    demands_arr = None
    capacities_arr = None
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(-1, 1)
        if capacities_arr.ndim == 0:
            capacities_arr = capacities_arr.reshape(1)
        if demands_arr.shape[0] == n and capacities_arr.shape[0] > 0:
            num_resources = capacities_arr.shape[0]
            safe_cap = np.where(capacities_arr > 0, capacities_arr, 1.0)
            scarcity = np.sum(demands_arr / safe_cap, axis=1)
    
    # Successor count
    succ_count = np.array([len(s) for s in succs], dtype=float)
    
    # Resource contention: for each activity, sum over resources of
    # (demand on that resource) * (number of successors that also demand that resource)
    contention = np.zeros(n, dtype=float)
    if demands_arr is not None and capacities_arr is not None and num_resources > 0:
        for j in range(n):
            for v in succs[j]:
                # Sum over resources where both j and v have non-zero demand
                shared = np.minimum(demands_arr[j], demands_arr[v])
                contention[j] += np.sum(shared)
    
    # NEW: Resource Bottleneck Centrality
    # For each activity, compute the sum over all downstream activities of
    # (1/capacity) weighted by the number of downstream activities that depend on it
    # This identifies activities that gate the most resource-constrained downstream work
    
    # First, compute the set of all descendants for each activity
    # Use reverse topological order to compute descendant counts efficiently
    descendant_count = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            descendant_count[u] += 1 + descendant_count[v]
    
    # Resource bottleneck centrality: for each activity u, sum over all descendants d
    # of (scarcity of d) * (1 / (distance from u to d))
    # Approximation: use scarcity of direct and indirect successors weighted by inverse capacity
    bottleneck_centrality = np.zeros(n, dtype=float)
    if demands_arr is not None and capacities_arr is not None and num_resources > 0:
        safe_cap = np.where(capacities_arr > 0, capacities_arr, 1.0)
        inv_cap = 1.0 / safe_cap  # shape: (num_resources,)
        
        # For each activity, compute the sum of scarcity over all its descendants
        # weighted by the inverse capacity of the most constrained resource
        # Simplified: for each activity, sum scarcity of all descendants
        # and weight by the minimum capacity (most constrained resource)
        
        # Compute for each activity the total downstream scarcity
        # Process in reverse topological order
        downstream_scarcity = np.zeros(n, dtype=float)
        for u in reversed(topo_order):
            for v in succs[u]:
                downstream_scarcity[u] += scarcity[v] + downstream_scarcity[v]
        
        # Combine with inverse capacity: activities that free up scarce resources
        # should be prioritized. Use the minimum capacity as a proxy for scarcity pressure
        min_inv_cap = np.min(inv_cap) if num_resources > 0 else 1.0
        bottleneck_centrality = downstream_scarcity * min_inv_cap
    
    # Normalize
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    tf_norm = normalize(-total_float)
    sc_norm = normalize(scarcity)
    sc_count_norm = normalize(succ_count)
    cont_norm = normalize(contention)
    bc_norm = normalize(bottleneck_centrality)
    
    # Weights: emphasize critical path, then resource bottleneck centrality, 
    # then resource contention, then scarcity, then successor count
    w_tf = 0.45
    w_bc = 0.25
    w_cont = 0.15
    w_sc = 0.10
    w_succ = 0.05
    
    priority_scores = (w_tf * tf_norm + 
                       w_bc * bc_norm + 
                       w_cont * cont_norm + 
                       w_sc * sc_norm + 
                       w_succ * sc_count_norm)
    
    # Check time
    if time.time() - t0 > _TIMEOUT - 2.0:
        # Return what we have
        return priority_scores
    
    return priority_scores
