"""Heuristic for ACO-CVRP: Delaunay-Constrained Adaptive Parameter Selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

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
    
    # We need to triangulate all points including the depot.
    # scipy.spatial.Delaunay works on points.
    try:
        tri = Delaunay(coordinates)
        for simplex in tri.simplices:
            # Each simplex is a triangle of 3 points
            i, j, k = simplex
            # Set edges between these points
            mask[i, j] = True
            mask[j, i] = True
            mask[j, k] = True
            mask[k, j] = True
            mask[i, k] = True
            mask[k, i] = True
    except Exception:
        # Fallback: if triangulation fails (e.g., collinear points), 
        # use a k-nearest neighbor graph as a sparse alternative
        # Or just return all True (dense) to be safe, but that defeats the purpose.
        # Let's try a simple KDTree approach for robustness if Delaunay fails
        from scipy.spatial import cKDTree
        try:
            tree = cKDTree(coordinates)
            # 5 nearest neighbors is a reasonable sparse graph
            _, indices = tree.query(coordinates, k=5)
            for i in range(n):
                for j in indices[i]:
                    mask[i, j] = True
        except Exception:
            # Absolute fallback: dense
            mask.fill(True)
    
    # Ensure diagonal is False (no self-loops in heuristic relevance)
    np.fill_diagonal(mask, False)
    
    return mask

def _fast_aco_sim(distance_matrix, demands, capacity, alpha, beta, delaunay_mask, n_ants=10, n_iter=5):
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
    # savings[i,j] = d[i,0] + d[0,j] - d[i,j]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    # Normalize savings to be positive and comparable
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_heur = np.ones_like(savings)
    else:
        sav_heur = savings_pos / max_sav
    
    # Combine
    # H = d_inv^alpha * sav_heur^beta
    # Use logs to avoid under/overflow
    log_h = alpha * np.log(d_inv + 1e-9) + beta * np.log(sav_heur + 1e-9)
    # Exponentiate back, normalize to keep scale reasonable
    h = np.exp(log_h)
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
    
    # Apply Delaunay constraint:
    # Edges NOT in the Delaunay graph get a very low heuristic value.
    # This encourages the ACO to stay within the planar graph.
    # We use a small epsilon instead of 0 to avoid log(0) issues in the actual run,
    # but here we can just mask it out strongly.
    # The task says "Heuristic entries that are not positive become 1e-9".
    # So we set non-Delaunay edges to a very small value, e.g., 1e-6 relative to max,
    # or just 1e-9 if we want strict enforcement.
    # Let's use 1e-9 for non-Delaunay edges to effectively forbid them.
    h[~delaunay_mask] = 1e-9
    
    np.fill_diagonal(h, 0.0)
    
    # Pheromone initialization
    P = np.ones((n, n))
    
    best_cost = float('inf')
    
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
                # Optimization: Precompute feasible mask based on capacity?
                # For speed in sim, we loop.
                feasible = []
                for j in range(1, n):
                    if not visited[j] and demands[j] <= remaining_cap + 1e-9:
                        feasible.append(j)
                
                if not feasible:
                    # Return to depot
                    cost += distance_matrix[node, 0]
                    break
                
                # Calculate probabilities: P * H
                # We must be careful: if all feasible nodes have H=1e-9, prob is tiny.
                # But since we are simulating, we just proceed.
                
                probs = np.zeros(len(feasible))
                for k, j in enumerate(feasible):
                    probs[k] = P[node, j] * h[node, j]
                
                total_prob = np.sum(probs)
                if total_prob <= 1e-9:
                    # Fallback to random among feasible
                    idx = np.random.randint(len(feasible))
                else:
                    probs /= total_prob
                    r = np.random.random()
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
        
        # Update Pheromone (Simplified: Evaporate and Add)
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
        # Fallback to simple distance heuristic
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # If time is running out, return a safe default
    if _remaining() < 20.0:
        d = dm.copy()
        np.fill_diagonal(d, 1e9)
        return 1.0 / (d + 1e-9)
    
    # Build the Delaunay mask
    try:
        delaunay_mask = _build_delaunay_mask(coords)
    except Exception:
        # If we can't build the mask, fall back to dense (all True)
        delaunay_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(delaunay_mask, False)
    
    # Grid search over alpha (distance weight) and beta (savings weight)
    # Typical good ranges: alpha in [1, 3], beta in [0, 2]
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
                pass
        if _remaining() < 30.0:
            break
    
    # Refine around the best found parameters
    if _remaining() > 60.0:
        ref_alphas = np.linspace(max(0.5, best_alpha - 0.5), best_alpha + 0.5, 5)
        ref_betas = np.linspace(max(0.0, best_beta - 0.5), best_beta + 0.5, 5)
        for a in ref_alphas:
            if a <= 0: continue
            for b in ref_betas:
                if b < 0: continue
                if _remaining() < 40.0:
                    break
                try:
                    score = _fast_aco_sim(dm, D, cap, a, b, delaunay_mask, n_ants=10, n_iter=5)
                    if score < best_score:
                        best_score = score
                        best_alpha = a
                        best_beta = b
                except Exception:
                    pass
            if _remaining() < 40.0:
                break
    
    # Construct the final heuristic matrix with the best parameters
    d = dm.copy()
    np.fill_diagonal(d, 1e9)
    
    d_inv = 1.0 / (d + 1e-9)
    
    d_i0 = dm[:, 0]
    d_j0 = dm[0, :]
    savings = d_i0[:, np.newaxis] + d_j0[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    max_sav = np.max(savings_pos)
    if max_sav < 1e-9:
        sav_heur = np.ones_like(savings)
    else:
        sav_heur = savings_pos / max_sav
    
    log_h = best_alpha * np.log(d_inv + 1e-9) + best_beta * np.log(sav_heur + 1e-9)
    h = np.exp(log_h)
    
    # Normalize to prevent numerical issues in the actual ACO run
    max_h = np.max(h)
    if max_h > 1e-9:
        h = h / max_h
    else:
        h = np.ones_like(h)
        
    # Apply Delaunay constraint to the final heuristic
    # Edges not in the Delaunay graph are set to a very low value.
    # This forces the ACO to prefer edges in the planar graph.
    h[~delaunay_mask] = 1e-9
    
    # Ensure positivity and handle diagonal
    h = np.maximum(h, 1e-9)
    np.fill_diagonal(h, 0.0)
    
    return h
# EVOLVE-BLOCK-END
