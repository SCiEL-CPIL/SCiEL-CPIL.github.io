"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 130.0
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: Dynamic resource-contention priority combining:
    1. Bottom Level (longest path to end)
    2. Max resource utilization ratio among unscheduled activities sharing resources
    3. Total Float (slack)
    4. Duration
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Emergency fallback
        durations = np.asarray(instance["durations"], dtype=float)
        if len(durations) == 0: return np.array([])
        return durations / (np.max(durations) + 1e-9)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # 1. Build graph
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(top_order) < n:
        top_order = list(range(n))

    # 2. Compute Bottom Level (Longest path from node to end)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            max_succ_bl = np.max([bottom_level[v] for v in successors[u]])
        else:
            max_succ_bl = 0.0
        bottom_level[u] = durations[u] + max_succ_bl

    # 3. Compute Resource Pressure and Demand Matrix
    rp = np.zeros(n)
    n_res = 0
    d_matrix = np.zeros((n, 1), dtype=float)
    capacities_arr = np.array([1.0])
    
    if demands is not None and capacities is not None:
        capacities_arr = np.asarray(capacities, dtype=float)
        if len(capacities_arr) == 0:
            capacities_arr = np.array([1.0])
        else:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)
            
        n_res = len(capacities_arr)
        d_matrix = np.zeros((n, n_res), dtype=float)
        
        if isinstance(demands, list):
            for i in range(n):
                if i < len(demands):
                    d_i = demands[i]
                    if isinstance(d_i, dict):
                        for r, val in d_i.items():
                            if r < n_res:
                                d_matrix[i, r] = val
                    elif isinstance(d_i, (list, tuple, np.ndarray)):
                        d_arr = np.asarray(d_i, dtype=float)
                        if d_arr.ndim == 1:
                            min_len = min(len(d_arr), n_res)
                            d_matrix[i, :min_len] = d_arr[:min_len]
                    else:
                        if n_res > 0:
                            d_matrix[i, 0] = float(d_i)
        else:
            try:
                d_mat_full = np.asarray(demands, dtype=float)
                if d_mat_full.shape[0] == n:
                    if d_mat_full.ndim == 1:
                         d_matrix[:, 0] = d_mat_full
                    else:
                        min_res = min(d_mat_full.shape[1], n_res)
                        d_matrix[:, :min_res] = d_mat_full[:, :min_res]
            except:
                pass

        # Calculate pressure: sum(d / c)
        if n_res > 0:
            ratios = d_matrix / capacities_arr
            rp = np.sum(ratios, axis=1)

    # 4. Compute Total Float
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in top_order:
        if pred_lists[u]:
            es[u] = np.max([ef[p] for p in pred_lists[u]])
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]
        
    cp_length = np.max(ef) if n > 0 else 0.0
    
    ls = np.zeros(n)
    lf = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            lf[u] = np.min([ls[v] for v in successors[u]])
        else:
            lf[u] = cp_length
        ls[u] = lf[u] - durations[u]
        
    tf = ls - es

    # 5. Dynamic Resource-Contention Priority
    # For each activity, compute the max resource utilization ratio among
    # all other activities that share any resource with it
    # This forces high-demand activities to be prioritized early
    
    contention = np.zeros(n)
    if n_res > 0:
        # For each activity i, find max(d_j/r / c_r) for j != i sharing resource r with i
        for i in range(n):
            max_ratio = 0.0
            for r in range(n_res):
                if d_matrix[i, r] > 0:
                    # Find all activities j that also use resource r
                    for j in range(n):
                        if j == i:
                            continue
                        if d_matrix[j, r] > 0:
                            ratio = d_matrix[j, r] / capacities_arr[r]
                            if ratio > max_ratio:
                                max_ratio = ratio
            contention[i] = max_ratio

    # 6. Normalize Metrics
    # Bottom Level
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    bl_min = np.min(bottom_level)
    bl_range = bl_max - bl_min
    if bl_range > 0:
        bl_norm = (bottom_level - bl_min) / bl_range
    else:
        bl_norm = np.ones(n)

    # Resource Pressure
    rp_max = np.max(rp) if np.max(rp) > 0 else 1.0
    rp_min = np.min(rp)
    rp_range = rp_max - rp_min
    if rp_range > 0:
        rp_norm = (rp - rp_min) / rp_range
    else:
        rp_norm = np.ones(n)

    # Duration
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_min = np.min(durations)
    dur_range = dur_max - dur_min
    if dur_range > 0:
        dur_norm = (durations - dur_min) / dur_range
    else:
        dur_norm = np.ones(n)

    # Contention
    cont_max = np.max(contention) if np.max(contention) > 0 else 1.0
    cont_min = np.min(contention)
    cont_range = cont_max - cont_min
    if cont_range > 0:
        cont_norm = (contention - cont_min) / cont_range
    else:
        cont_norm = np.zeros(n)

    # Total Float
    tf_max = np.max(tf) if np.max(tf) > 0 else 1.0
    tf_min = np.min(tf)
    tf_range = tf_max - tf_min
    if tf_range > 0:
        tf_norm = (tf - tf_min) / tf_range
    else:
        tf_norm = np.zeros(n)

    # 7. Combine Scores
    # Key insight: Prioritize activities that are:
    # 1. On critical path (high BL, low TF)
    # 2. High resource demand relative to capacity
    # 3. Cause high contention with other activities
    # 4. Long duration (occupies resources longer)
    
    criticality = bl_norm * (1.0 - tf_norm)
    resource_demand = rp_norm
    contention_score = cont_norm
    duration_score = dur_norm
    
    # Interaction: critical AND contentious
    interaction = criticality * contention_score
    
    # Final score: weighted combination
    w_crit = 0.30
    w_res = 0.25
    w_cont = 0.20
    w_dur = 0.10
    w_inter = 0.15
    
    scores = (w_crit * criticality + w_res * resource_demand + 
              w_cont * contention_score + w_dur * duration_score + 
              w_inter * interaction)

    # Ensure scores are in a reasonable range
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
