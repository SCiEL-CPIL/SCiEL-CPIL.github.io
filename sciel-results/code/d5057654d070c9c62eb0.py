"""Heuristic for ACO-CVRP: Delaunay-Constrained Adaptive Parameter Selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree

_T0 = time.time()
_BUDGET = 330.0
_DEADLINE = _T0 + _BUDGET
_TIME_MARGIN = 15.0

def _remaining():
    return _DEADLINE - _TIME_MARGIN - time.time()

def _build_delaunay_mask(coordinates):
    """
    Build a boolean mask of edges present in the Delaunay triangulation.
    Node 0 is the depot.
    """
    n = len(coordinates)
    mask = np.zeros((n, n), dtype=bool)
    
    try:
        tri = Delaunay(coordinates)
        for simplex in tri.simplices:
            i, j, k = simplex
            mask[i, j] = True
            mask[j, i] = True
            mask[j, k] = True
            mask[k, j] = True
            mask[i, k] = True
            mask[k, i] = True
    except Exception:
        # Fallback: if triangulation fails, use k-nearest neighbors
        try:
            tree = cKDTree(coordinates)
            _, indices = tree.query(coordinates, k=5)
            for i in range(n):
                for j in indices[i]:
                    mask[i, j] = True
        except Exception:
            # Absolute fallback: dense
            mask.fill(True)
    
    # Ensure diagonal is False
    np.fill_diagonal(mask, False)
    
    return mask

def _compute_savings_heuristic(distance_matrix):
    """
    Compute the savings heuristic for CVRP.
    savings[i,j] = d[i,0] + d[0,j] - d[i,j]
    """
    n = distance_matrix.shape[0]
    d_i0 = distance_matrix[:, 0]
    d_j0 = distance_matrix[0, :]
    
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - distance_matrix
    
    # Clip negative savings to 0
    savings_pos = np.maximum(savings, 0.0)
    
    # Normalize to [0, 1]
    max_sav = np.max(savings_pos)
    if max_sav > 1e-9:
        savings_norm = savings_pos / max_sav
    else:
        savings_norm = np.ones_like(savings_pos)
        
    # Zero out diagonal
    np.fill_diagonal(savings_norm, 0.0)
    
    return savings_norm

def _build_savings_aware_delaunay_mask(coordinates, distance_matrix, savings_heuristic, k_nearest=3):
    """
    Build a mask that includes Delaunay edges AND high-savings edges.
    """
    n = len(coordinates)
    mask = np.zeros((n, n), dtype=bool)
    
    # 1. Delaunay Edges
    try:
        tri = Delaunay(coordinates)
        for simplex in tri.simplices:
            i, j, k = simplex
            mask[i, j] = True; mask[j, i] = True
            mask[j, k] = True; mask[k, j] = True
            mask[i, k] = True; mask[k, i] = True
    except Exception:
        try:
            tree = cKDTree(coordinates)
            _, indices = tree.query(coordinates, k=k_nearest)
            for i in range(n):
                for j in indices[i]:
                    mask[i, j] = True
        except Exception:
            mask.fill(True)
            np.fill_diagonal(mask, False)
            return mask

    # 2. High Savings Edges
    # Edges with savings in the top 10% are likely to be good for reducing tour count
    threshold = np.percentile(savings_heuristic, 90)
    high_savings_mask = savings_heuristic > threshold
    
    # Combine: Delaunay OR High Savings
    mask = mask | high_savings_mask
    
    # Ensure symmetry
    mask = mask | mask.T
    
    # Zero diagonal
    np.fill_diagonal(mask, False)
    
    return mask

def _fast_aco_sim(distance_matrix, demands, capacity, alpha, beta, delaunay_mask, n_ants=10, n_iter=5, seed=42):
    """
    Run a fast, simplified ACO simulation to evaluate heuristic parameters.
    Returns the best cost found.
    """
    n = distance_matrix.shape[0]
    # Precompute heuristic matrix for these params
    d = distance_matrix.copy()
    np.fill_diagonal(d, 1e9)
    
    # Base distance heuristic
    d_inv = 1.0 / (d + 1e-9)
    
    # Savings heuristic
    d_i0 = distance_matrix[:, 0]
    d_j0 = distance_matrix[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_heur = np.ones_like(savings)
    else:
        sav_heur = savings_pos / max_sav
    
    # Combine
    log_h = alpha * np.log(d_inv + 1e-9) + beta * np.log(sav_heur + 1e-9)
    h = np.exp(log_h)
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
    
    # Apply Delaunay constraint
    h[~delaunay_mask] = 1e-9
    np.fill_diagonal(h, 0.0)
    
    # Pheromone initialization
    P = np.ones((n, n))
    
    best_cost = float('inf')
    rng = np.random.default_rng(seed)
    
    for _ in range(n_iter):
        iteration_costs = []
        for _ in range(n_ants):
            node = 0
            remaining_cap = capacity
            cost = 0.0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            
            while True:
                # Find feasible neighbors
                feasible = []
                for j in range(1, n):
                    if not visited[j] and demands[j] <= remaining_cap + 1e-9:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    cost += distance_matrix[node, 0]
                    break
                
                # Calculate probabilities
                probs = np.zeros(len(feasible))
                for k, j in enumerate(feasible):
                    probs[k] = P[node, j] * h[node, j]
                
                total_prob = np.sum(probs)
                if total_prob <= 1e-9:
                    idx = rng.integers(len(feasible))
                else:
                    probs /= total_prob
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    idx = np.searchsorted(cumsum, r)
                    if idx >= len(feasible):
                        idx = len(feasible) - 1
                
                next_node = feasible[idx]
                cost += distance_matrix[node, next_node]
                remaining_cap -= demands[next_node]
                visited[next_node] = True
                node = next_node
            
        iteration_costs.append(cost)
        
        best_iter = min(iteration_costs)
        if best_iter < best_cost:
            best_cost = best_iter
        
        # Update Pheromone
        P *= 0.9
        P += 1.0 / best_iter
        
    return best_cost

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    Uses Delaunay triangulation to constrain the search space and
    adaptive parameter selection via local simulation.
    """
    dm = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    D = np.asarray(demands, dtype=float)
    cap = float(capacity)
    n = dm.shape[0]
    
    # Basic validation
    if dm.shape != (n, n) or not np.isfinite(dm).all() or not np.isfinite(D).all():
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # If time is running out, return a safe default
    if _remaining() < 20.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Build the savings-aware Delaunay mask
    savings_norm = _compute_savings_heuristic(dm)
    try:
        delaunay_mask = _build_savings_aware_delaunay_mask(coords, dm, savings_norm)
    except Exception:
        delaunay_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(delaunay_mask, False)
    
    # Grid search over alpha (distance weight) and beta (savings weight)
    alphas = [1.0, 1.5, 2.0, 2.5, 3.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    
    best_score = float('inf')
    best_alpha = 2.0
    best_beta = 0.5
    
    for a in alphas:
        for b in betas:
            if _remaining() < 30.0:
                break
            try:
                score = _fast_aco_sim(dm, D, cap, a, b, delaunay_mask, n_ants=10, n_iter=5)
                if score < best_score:
                    best_score = score
                    best_alpha = a
                    best_beta = b
            except Exception:
                continue
        if _remaining() < 30.0:
            break
    
    # Compute the final heuristic with the best parameters
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    d_inv = 1.0 / (d + 1e-9)
    
    log_h = best_alpha * np.log(d_inv + 1e-9) + best_beta * np.log(savings_norm + 1e-6)
    h = np.exp(log_h)
    
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
        
    # Apply mask
    h[~delaunay_mask] = 1e-9
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
