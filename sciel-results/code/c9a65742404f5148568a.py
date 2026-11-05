"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_TIME_LIMIT = 145.0
_START_TIME = None

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic: Weighted combination of Bottom Level (criticality),
    Resource Pressure (contention), and Duration.
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
    # BL[i] = duration[i] + max(BL[j] for j in successors[i])
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        if successors[u]:
            max_succ_bl = np.max([bottom_level[v] for v in successors[u]])
        else:
            max_succ_bl = 0.0
        bottom_level[u] = durations[u] + max_succ_bl

    # 3. Compute Resource Pressure
    # Sum of (demand / capacity) for each resource
    rp = np.zeros(n)
    if demands is not None and capacities is not None:
        capacities_arr = np.asarray(capacities, dtype=float)
        if len(capacities_arr) == 0:
            capacities_arr = np.array([1.0])
        else:
            capacities_arr = np.where(capacities_arr > 0, capacities_arr, 1.0)
            
        # Process demands
        n_res = len(capacities_arr)
        # Initialize demand matrix
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
                        # Scalar?
                        if n_res > 0:
                            d_matrix[i, 0] = float(d_i)
        else:
            # Try converting whole thing to array
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
        # d_matrix shape (n, n_res), capacities shape (n_res,)
        ratios = d_matrix / capacities_arr
        rp = np.sum(ratios, axis=1)

    # 4. Normalize Metrics
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

    # 5. Combine Scores
    # Weights: Bottom Level is the strongest indicator of critical path involvement.
    # Resource Pressure helps handle bottlenecks.
    # Duration is a weak tie-breaker.
    w_bl = 0.65
    w_rp = 0.25
    w_dur = 0.10

    # Soft-max penalty: penalize activities that consume a large fraction of scarce resources.
    # This keeps the score smooth and differentiable-like, and biases away from bottleneck
    # activities unless their criticality is high enough to justify them.
    soft_max_penalty = np.zeros(n)
    if demands is not None and capacities is not None and n_res > 0:
        # ratios[i, r] = demand / capacity
        # For scarce resources, large ratios should incur a nonlinear penalty.
        # Use a soft-max over resources to capture the worst resource contention.
        # log-sum-exp with temperature controls smoothness.
        temperature = 0.5
        # Clip to avoid overflow in exp
        ratios_clipped = np.clip(ratios, -50.0, 50.0)
        # Soft-max: log(sum(exp(ratios/temperature)))
        scaled = ratios_clipped / temperature
        max_scaled = np.max(scaled, axis=1, keepdims=True)
        # Handle all -inf (shouldn't happen due to clipping, but safe)
        max_scaled_safe = np.where(np.isfinite(max_scaled), max_scaled, 0.0)
        exp_scaled = np.exp(scaled - max_scaled_safe)
        log_sum_exp = max_scaled_safe + np.log(np.sum(exp_scaled, axis=1, keepdims=True))
        # Penalty is the soft-max value (higher = more contention)
        soft_max_penalty = log_sum_exp.squeeze(axis=1)
        # Normalize penalty to [0, 1]
        sm_max = np.max(soft_max_penalty) if np.max(soft_max_penalty) > 0 else 1.0
        sm_min = np.min(soft_max_penalty)
        sm_range = sm_max - sm_min
        if sm_range > 0:
            sm_norm = (soft_max_penalty - sm_min) / sm_range
        else:
            sm_norm = np.zeros(n)
    else:
        sm_norm = np.zeros(n)

    # Combine: criticality + duration - contention penalty
    # The penalty reduces priority for activities that are resource-intensive
    scores = w_bl * bl_norm + w_dur * dur_norm - w_rp * sm_norm

    # Ensure scores are in a reasonable range and non-negative for stability
    # (SGS likely uses higher = higher priority, so negative values are okay
    #  but let's keep them bounded)
    score_max = np.max(scores) if np.max(scores) > 0 else 1.0
    score_min = np.min(scores)
    score_range = score_max - score_min
    if score_range > 0:
        scores = (scores - score_min) / score_range
    else:
        scores = np.ones(n)

    return scores
# EVOLVE-BLOCK-END
