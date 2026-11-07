"""Heuristic for ACO-CVRP: Enhanced Geometric Heuristic with Local Density and Delaunay Constraints."""
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree

_T0 = time.time()
_BUDGET = 325.0  # Leave 5s safety margin
_DEADLINE = _T0 + _BUDGET

def _remaining():
    return max(0.0, _DEADLINE - time.time())

def _build_delaunay_mask(coords, n):
    """
    Build a boolean mask of edges from Delaunay triangulation.
    """
    mask = np.zeros((n, n), dtype=bool)
    try:
        tri = Delaunay(coords)
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(len(simplex)):
                    if i != j:
                        mask[simplex[i], simplex[j]] = True
    except Exception:
        # Fallback: k-nearest neighbors
        try:
            tree = cKDTree(coords)
            k = min(8, n - 1)
            distances, indices = tree.query(coords, k=k+1)
            for i in range(n):
                for j in indices[i]:
                    if j != i:
                        mask[i, j] = True
        except Exception:
            pass
    return mask

def _compute_local_scales(coords, n, k=5):
    """
    Compute local scale for each node based on k-nearest-neighbor distances.
    """
    try:
        tree = cKDTree(coords)
        k_eff = min(k, n - 1)
        distances, _ = tree.query(coords, k=k_eff + 1)
        # distances[:, 0] is 0 (self)
        local_scales = np.mean(distances[:, 1:], axis=1)
        local_scales = np.maximum(local_scales, 1e-6)
        return local_scales
    except Exception:
        return np.ones(n) * 0.1

def _run_aco_simulation(distance_matrix, demands, capacity, heuristic, n_ants=10, n_iter=5, seed=42):
    """
    Run a fast ACO simulation to evaluate a heuristic.
    Returns the best total route distance found by any ant.
    """
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    
    # Pheromone starts at 1
    pheromone = np.ones((n, n))
    
    best_cost_ever = float('inf')
    
    for iter_i in range(n_iter):
        iteration_costs = []
        
        for ant_i in range(n_ants):
            node = 0  # Start at depot
            remaining_load = capacity
            cost = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                # Find all unvisited customers that fit in remaining load
                feasible = []
                for j in range(1, n):
                    if not visited[j] and demands[j] <= remaining_load + 1e-9:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot and reload
                    cost += distance_matrix[node, 0]
                    break
                
                # Calculate probabilities: pheromone[i,j] * heuristic[i,j]
                # Vectorized probability calculation for speed
                probs = np.array([pheromone[node, j] * heuristic[node, j] for j in feasible])
                total_prob = np.sum(probs)
                
                if total_prob <= 1e-15:
                    # Fallback: uniform random
                    idx = np.random.randint(len(feasible))
                else:
                    probs = probs / total_prob
                    r = np.random.random()
                    cumulative = np.cumsum(probs)
                    idx = np.searchsorted(cumulative, r)
                    if idx >= len(feasible):
                        idx = len(feasible) - 1
                
                next_node = feasible[idx]
                cost += distance_matrix[node, next_node]
                remaining_load -= demands[next_node]
                visited[next_node] = True
                node = next_node
        
        iteration_costs.append(cost)
        
        best_this_iter = min(iteration_costs)
        if best_this_iter < best_cost_ever:
            best_cost_ever = best_this_iter
        
        # Pheromone update: evaporate then add
        pheromone *= 0.9
        pheromone += 1.0 / best_this_iter
    
    return best_cost_ever

def _construct_heuristic(distance_matrix, coordinates, demands, capacity, params, delaunay_mask=None, local_scales=None):
    """
    Construct a heuristic matrix from parameters.
    """
    dm = distance_matrix
    n = dm.shape[0]
    
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    
    # Inverse distance component
    w_dist = max(params.get('w_dist', 1.0), 0.0)
    gamma = max(params.get('gamma', 2.0), 0.0)
    dist_inv = 1.0 / (d + 1e-9)
    dist_comp = dist_inv ** gamma if gamma > 0 else dist_inv
    
    # Savings component: d[i,0] + d[0,j] - d[i,j]
    w_sav = max(params.get('w_sav', 0.5), 0.0)
    delta = max(params.get('delta', 1.0), 0.0)
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_norm = np.ones_like(savings)
    else:
        sav_norm = savings_pos / max_sav
    sav_comp = sav_norm ** delta if delta > 0 else sav_norm
    
    # Local density component
    # Boost edges that are short relative to local scale and in Delaunay
    w_loc = max(params.get('w_loc', 1.0), 0.0)
    if local_scales is not None and w_loc > 0:
        ls_max = np.maximum(local_scales[:, np.newaxis], local_scales[np.newaxis, :])
        # Ratio of distance to local scale
        ratio = d / (ls_max + 1e-9)
        # Base local score: 1 if ratio is small, decays as ratio increases
        # Using a smooth decay: exp(-ratio * decay_factor)
        decay = max(params.get('loc_decay', 2.0), 0.1)
        local_score = np.exp(-ratio * decay)
        
        # Apply Delaunay mask: only boost if in Delaunay
        if delaunay_mask is not None:
            local_score = np.where(delaunay_mask, local_score, 1.0)
        else:
            local_score = np.where(delaunay_mask, local_score, 1.0) if delaunay_mask is not None else local_score
            
        # Combine with weight: 1 + w_loc * (local_score - 1)
        # This makes local_score=1 neutral, local_score<1 penalized, local_score>1 boosted
        # But since local_score <= 1 always (exp), we want to boost high local_score.
        # So: 1 + w_loc * local_score
        loc_comp = 1.0 + w_loc * local_score
    else:
        loc_comp = np.ones_like(dm)
    
    # Combine multiplicatively in log space for stability
    log_h = (w_dist * np.log(dist_comp + 1e-12) +
             w_sav * np.log(sav_comp + 1e-12) +
             np.log(loc_comp + 1e-12))
    h = np.exp(log_h)
    
    # Normalize
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
    
    # Ensure all entries are positive
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 0.0)
    
    return h

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic validation and fallback
    if dm.shape != (n, n) or not np.isfinite(dm).all() or not np.isfinite(D).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Build Delaunay mask and local scales
    try:
        delaunay_mask = _build_delaunay_mask(coords, n)
    except Exception:
        delaunay_mask = None
        
    try:
        local_scales = _compute_local_scales(coords, n, k=5)
    except Exception:
        local_scales = None
        
    # Check time
    if _remaining() < 10.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Initialize with a reasonable default
    default_params = {
        'w_dist': 2.0,
        'gamma': 2.0,
        'w_sav': 1.0,
        'delta': 1.0,
        'w_loc': 1.0,
        'loc_decay': 2.0
    }
    
    best_params = default_params.copy()
    best_score = float('inf')
    
    # Grid search
    # We reduced the dimensionality by removing demand component which seemed less impactful
    # and focusing on distance, savings, and local density.
    grid = {
        'w_dist': [1.0, 2.0, 3.0],
        'gamma': [1.5, 2.0, 2.5],
        'w_sav': [0.5, 1.0, 2.0],
        'delta': [1.0],
        'w_loc': [0.5, 1.0, 2.0],
        'loc_decay': [1.0, 2.0, 3.0]
    }
    
    # Evaluate a subset of the grid
    for w_dist in grid['w_dist']:
        if _remaining() < 15.0: break
        for gamma in grid['gamma']:
            if _remaining() < 15.0: break
            for w_sav in grid['w_sav']:
                if _remaining() < 15.0: break
                for delta in grid['delta']:
                    if _remaining() < 15.0: break
                    for w_loc in grid['w_loc']:
                        if _remaining() < 15.0: break
                        for loc_decay in grid['loc_decay']:
                            if _remaining() < 15.0: break
                            params = {
                                'w_dist': w_dist,
                                'gamma': gamma,
                                'w_sav': w_sav,
                                'delta': delta,
                                'w_loc': w_loc,
                                'loc_decay': loc_decay
                            }
                            try:
                                h = _construct_heuristic(dm, coords, D, cap, params, delaunay_mask, local_scales)
                                score = _run_aco_simulation(dm, D, cap, h, n_ants=10, n_iter=5, seed=42)
                                if score < best_score:
                                    best_score = score
                                    best_params = params.copy()
                            except Exception:
                                pass
    
    # Fine-tuning: local search around best params
    if _remaining() > 30.0:
        step_mult = 0.2
        for _iter in range(5):
            if _remaining() < 20.0:
                break
            improved = False
            for key in ['w_dist', 'gamma', 'w_sav', 'delta', 'w_loc', 'loc_decay']:
                if _remaining() < 20.0:
                    break
                for direction in [-step_mult, step_mult]:
                    new_params = best_params.copy()
                    new_val = best_params[key] + direction
                    if new_val < 0.1:
                        new_val = 0.1
                    new_params[key] = new_val
                    try:
                        h = _construct_heuristic(dm, coords, D, cap, new_params, delaunay_mask, local_scales)
                        score = _run_aco_simulation(dm, D, cap, h, n_ants=10, n_iter=5, seed=42)
                        if score < best_score:
                            best_score = score
                            best_params = new_params
                            improved = True
                            break
                    except Exception:
                        pass
                if improved:
                    break
            if not improved:
                step_mult *= 0.5
                if step_mult < 0.05:
                    break
    
    # Construct final heuristic with best parameters
    final_h = _construct_heuristic(dm, coords, D, cap, best_params, delaunay_mask, local_scales)
    
    # Final safety
    final_h = np.maximum(final_h, 1e-9)
    np.fill_diagonal(final_h, 0.0)
    
    return final_h
