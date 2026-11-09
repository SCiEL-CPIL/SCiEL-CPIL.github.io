"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def _compute_features(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Build successor and predecessor counts
    succ_counts = np.zeros(n, dtype=int)
    pred_counts = np.zeros(n, dtype=int)
    for i in range(n):
        preds = predecessors[i]
        pred_counts[i] = len(preds)
        for p in preds:
            succ_counts[p] += 1
            
    # In-degrees for topo sort
    in_degree = pred_counts.copy()
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg = in_degree.copy()
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        # Find successors
        for j in range(n):
            if node in predecessors[j]:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    
    # Longest path from node to end
    longest_path = np.zeros(n, dtype=float)
    for node in reversed(topo_order):
        max_succ_path = 0.0
        for j in range(n):
            if node in predecessors[j]:
                max_succ_path = max(max_succ_path, longest_path[j])
        longest_path[node] = times[node] + max_succ_path
        
    # Topo position (0 to 1, earlier is higher)
    pos_in_topo = np.zeros(n, dtype=float)
    for idx, node in enumerate(topo_order):
        pos_in_topo[node] = idx
    pos_norm = 1.0 - (pos_in_topo / (n - 1 + 1e-9))
    
    # Normalize all features to [0, 1]
    def norm(v):
        v = np.array(v, dtype=float)
        mn, mx = v.min(), v.max()
        if mx - mn < 1e-9:
            return np.ones_like(v)
        return (v - mn) / (mx - mn)
        
    f_time = norm(times)
    f_lp = norm(longest_path)
    f_succ = norm(succ_counts)
    f_pred = norm(pred_counts)
    f_pos = pos_norm
    
    return np.column_stack((f_time, f_lp, f_succ, f_pred, f_pos))

def _run_greedy(instance, scores):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    assigned = [False] * n
    stations = 1
    current_load = 0.0
    
    while sum(assigned) < n:
        eligible = []
        for i in range(n):
            if not assigned[i]:
                ok = True
                for p in predecessors[i]:
                    if not assigned[p]:
                        ok = False
                        break
                if ok:
                    eligible.append(i)
                    
        if not eligible:
            break
            
        # Sort by score descending
        eligible.sort(key=lambda x: scores[x], reverse=True)
        
        placed = False
        for i in eligible:
            if current_load + times[i] <= cycle_time:
                assigned[i] = True
                current_load += times[i]
                placed = True
                break
                
        if not placed:
            stations += 1
            current_load = 0.0
            
    return stations

def priority(instance):
    features = _compute_features(instance)
    n = features.shape[0]
    
    # Initial weights: based on the current program's logic
    # Current: 0.5 time, 0.3 lp, 0.1 succ, 0.0 pred, 0.1 pos
    w = np.array([0.5, 0.3, 0.1, 0.0, 0.1])
    
    best_w = w.copy()
    best_stations = _run_greedy(instance, features @ best_w)
    
    # Local Search to optimize weights for this specific instance
    for _ in range(200):
        # Perturb weights
        noise = np.random.normal(0, 0.1, size=5)
        w_new = best_w + noise
        
        scores = features @ w_new
        stations = _run_greedy(instance, scores)
        
        if stations < best_stations:
            best_w = w_new
            best_stations = stations
    
    return features @ best_w
# EVOLVE-BLOCK-END
