# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance, n):
    """Build predecessor/successor lists from instance."""
    preds = instance.get("predecessors", None)
    succs = [[] for _ in range(n)]
    pred_list = [[] for _ in range(n)]
    
    if preds is not None:
        for i in range(n):
            p = preds[i]
            if p is not None:
                if isinstance(p, (list, tuple, np.ndarray)):
                    for j in p:
                        j = int(j)
                        if 0 <= j < n:
                            pred_list[i].append(j)
                            succs[j].append(i)
                else:
                    j = int(p)
                    if 0 <= j < n:
                        pred_list[i].append(j)
                        succs[j].append(i)
    
    return pred_list, succs


def _critical_path_lengths(instance, n):
    """Compute longest path from each activity to the end (tail) and from start to each activity."""
    pred_list, succs = _build_graph(instance, n)
    durations = np.asarray(instance["durations"], dtype=float)
    
    # Topological sort using Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        for p in pred_list[i]:
            in_degree[i] += 1
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        # Use a simple list as queue (pop from end for stack-like, or use index)
        node = queue.pop(0)
        topo_order.append(node)
        for s in succs[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # If not all nodes in topo_order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        # Fallback: just use duration
        return durations.copy(), durations.copy()
    
    # Compute earliest start times (forward pass)
    est = np.zeros(n, dtype=float)
    for node in topo_order:
        if pred_list[node]:
            est[node] = max(est[p] + durations[p] for p in pred_list[node])
        else:
            est[node] = 0.0
    
    # Compute latest finish times (backward pass) - longest path from node to end
    lft = np.zeros(n, dtype=float)  # longest float time: longest path from node to any sink
    for node in reversed(topo_order):
        if succs[node]:
            lft[node] = max(durations[node] + lft[s] for s in succs[node])
        else:
            lft[node] = durations[node]
    
    return est, lft


def _resource_intensity(instance, n):
    """Compute resource demand intensity for each activity."""
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    intensity = np.zeros(n, dtype=float)
    
    if demands is not None and capacities is not None:
        demands_arr = np.atleast_2d(np.asarray(demands, dtype=float))
        caps_arr = np.asarray(capacities, dtype=float)
        
        # Handle case where demands is 1D (single resource) or 2D (multiple resources)
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(n, 1)
        
        if len(demands_arr) != n:
            return intensity
        
        # For each activity, compute max ratio of demand to capacity across resources
        # Also consider total demand pressure
        for i in range(n):
            d = demands_arr[i]
            # Normalize by capacity
            if len(d) == len(caps_arr) and np.all(caps_arr > 0):
                ratios = d / caps_arr
                intensity[i] = np.max(ratios)
            else:
                intensity[i] = np.sum(d)
    
    return intensity


def _resource_contention_score(instance, n):
    """
    Estimate resource contention: for each activity, how much it contributes
    to potential bottlenecks. Uses a soft-max over resource utilization ratios.
    """
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    durations = np.asarray(instance["durations"], dtype=float)
    
    score = np.zeros(n, dtype=float)
    
    if demands is not None and capacities is not None:
        demands_arr = np.atleast_2d(np.asarray(demands, dtype=float))
        caps_arr = np.asarray(capacities, dtype=float)
        
        if demands_arr.ndim == 1:
            demands_arr = demands_arr.reshape(n, 1)
        
        if len(demands_arr) != n:
            return score
        
        # For each activity, compute a contention score based on:
        # 1. Peak resource ratio (max demand/capacity)
        # 2. Weighted by duration (longer activities with high demand are worse)
        # 3. Soft-max over resources to capture multi-resource pressure
        
        for i in range(n):
            d = demands_arr[i]
            if len(d) == len(caps_arr) and np.all(caps_arr > 0):
                ratios = d / caps_arr
                # Soft-max with temperature to smooth
                temp = 2.0
                # Use log-sum-exp for numerical stability
                max_r = np.max(ratios)
                if max_r > 0:
                    exp_terms = np.exp((ratios - max_r) / temp)
                    soft_max = max_r + temp * np.log(np.sum(exp_terms))
                else:
                    soft_max = 0.0
                score[i] = soft_max * durations[i]
            else:
                score[i] = np.sum(d) * durations[i]
    
    return score


def _successor_weight(instance, n):
    """Compute weight based on number of successors (criticality)."""
    pred_list, succs = _build_graph(instance, n)
    
    weight = np.zeros(n, dtype=float)
    for i in range(n):
        weight[i] = len(succs[i])
    
    return weight


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Combines:
    1. Critical path length (tail) - activities on critical path first
    2. Resource contention - activities causing resource bottlenecks first
    3. Resource intensity - high demand/capacity ratio activities first
    4. Successor count - activities with many successors first
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute components
    est, lft = _critical_path_lengths(instance, n)
    
    if time.time() > deadline:
        return lft
    
    resource_int = _resource_intensity(instance, n)
    
    if time.time() > deadline:
        return lft + resource_int
    
    contention = _resource_contention_score(instance, n)
    
    if time.time() > deadline:
        return lft + resource_int + contention
    
    succ_weight = _successor_weight(instance, n)
    
    if time.time() > deadline:
        return lft + resource_int + contention + succ_weight
    
    # Normalize each component to [0, 1] range for stable combination
    def normalize(arr):
        arr = np.asarray(arr, dtype=float)
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_lft = normalize(lft)
    norm_res = normalize(resource_int)
    norm_cont = normalize(contention)
    norm_succ = normalize(succ_weight)
    norm_dur = normalize(durations)
    
    # Weighted combination
    # Critical path is most important, then resource contention, then others
    score = (
        0.40 * norm_lft +
        0.25 * norm_cont +
        0.15 * norm_res +
        0.10 * norm_succ +
        0.10 * norm_dur
    )
    
    return score
# EVOLVE-BLOCK-END
