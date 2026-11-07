"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _fast_aco_sim(dm, demands, capacity, H, n_ants=20, n_iterations=10, seed=42, time_limit=5.0):
    """
    Fast simulation of the evaluator's ACO.
    Uses reduced iterations (10) for faster candidate evaluation during heuristic selection.
    """
    start_time = time.time()
    n = dm.shape[0]
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.RandomState(seed)
    best_cost = np.inf
    demands = np.asarray(demands, dtype=np.float64)
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break

        iteration_costs = []

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break

            current_node = 0
            remaining_cap = capacity
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            total_dist = 0.0

            while True:
                valid_mask = ~visited
                valid_mask[0] = False
                valid_mask &= (demands <= remaining_cap)

                if not np.any(valid_mask):
                    total_dist += dm[current_node, 0]
                    remaining_cap = capacity
                    current_node = 0
                    if np.all(visited):
                        break
                    continue

                row_tau = tau[current_node]
                row_eta = H[current_node]

                weights = row_tau * row_eta
                weights[~valid_mask] = 0.0

                total_w = np.sum(weights)
                if total_w <= 0:
                    valid_indices = np.where(valid_mask)[0]
                    next_node = rng.choice(valid_indices)
                else:
                    probs = weights / total_w
                    next_node = rng.choice(n, p=probs)

                total_dist += dm[current_node, next_node]
                visited[next_node] = True
                remaining_cap -= demands[next_node]
                current_node = next_node

                if np.all(visited):
                    total_dist += dm[current_node, 0]
                    break

            iteration_costs.append(total_dist)

        if iteration_costs:
            current_best = min(iteration_costs)
            if current_best < best_cost:
                best_cost = current_best
            tau *= 0.9
            tau += 1.0 / current_best

    return best_cost


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Returns an edge-attractiveness matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Safe distance matrix for division
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # 1. Base Inverse Distance
    H_base = 1.0 / dm_safe
    
    # 2. Grid-based Spatial Clustering (from current program, known to be effective)
    grid_size = 4
    cell_x = (coords[:, 0] * grid_size).astype(int)
    cell_y = (coords[:, 1] * grid_size).astype(int)
    cell_x = np.clip(cell_x, 0, grid_size - 1)
    cell_y = np.clip(cell_y, 0, grid_size - 1)
    
    same_cell = (cell_x[:, None] == cell_x[None, :]) & (cell_y[:, None] == cell_y[None, :])
    H_grid = H_base * np.where(same_cell, 2.0, 0.5)
    
    # 3. Nearest Neighbor Heuristic (from inspiration)
    H_knn = np.zeros_like(dm)
    try:
        tree = cKDTree(coords)
        # Query 6 nearest neighbors (including self)
        knn_dists, knn_idxs = tree.query(coords, k=6)
        
        for i in range(n):
            for rank in range(1, 6): # Skip self (rank 0)
                j = knn_idxs[i, rank]
                d = knn_dists[i, rank]
                # Attractiveness based on rank and distance
                weight = (1.0 / (rank + 1)) * (1.0 / (1.0 + d))
                H_knn[i, j] = weight
                H_knn[j, i] = weight # Symmetrize
    except Exception:
        # Fallback if cKDTree fails for some reason
        H_knn = H_base.copy()
        
    # Ensure KNN matrix has no zeros where base has values, to avoid dead-ends if combined
    # Actually, we want the KNN structure to be distinct. 
    # But if we use it alone, zeros might be an issue if the graph is not fully connected in the heuristic sense?
    # The evaluator replaces non-positive with 1e-9. So zeros become 1e-9.
    # This effectively makes non-KNN edges very unattractive. This is a strong structural constraint.
    
    # 4. Demand-weighted Heuristic
    d_demands = demands.reshape(-1, 1)
    H_dem = 1.0 / (dm_safe * (1.0 + 0.1 * d_demands))
    
    # 5. Blends
    H_blend_knn_base = 0.5 * H_knn + 0.5 * H_base
    H_blend_grid_base = 0.5 * H_grid + 0.5 * H_base
    
    # List of candidates to evaluate
    candidates = [
        H_base,
        H_grid,
        H_knn,
        H_dem,
        H_blend_knn_base,
        H_blend_grid_base
    ]
    
    # Sanitize candidates: ensure finite and positive (or will be replaced by 1e-9 by evaluator)
    # The evaluator says: "Heuristic entries that are not positive become 1e-9"
    # So we just need to ensure no NaNs or Infs.
    sanitized_candidates = []
    for h in candidates:
        h_safe = np.where(np.isfinite(h), h, 1e-9)
        sanitized_candidates.append(h_safe)
        
    best_H = sanitized_candidates[0]
    best_score = np.inf
    
    # Evaluate candidates using fast ACO sim
    # Use 3 seeds for robustness
    seeds = [0, 1, 2]
    
    for i, h in enumerate(sanitized_candidates):
        if time.time() > deadline - 20.0:
            break
            
        costs = []
        for s in seeds:
            cost = _fast_aco_sim(dm, demands, capacity, h, 
                                  n_ants=20, n_iterations=10, 
                                  seed=s, time_limit=5.0)
            if cost < np.inf:
                costs.append(cost)
                
        if costs:
            avg_cost = np.mean(costs)
            if avg_cost < best_score:
                best_score = avg_cost
                best_H = h

    # If time remains, try to fine-tune the best candidate by blending with base
    if time.time() < deadline - 10.0:
        # Try blending the best found with the base inverse distance
        # This can sometimes smooth out overly aggressive structural heuristics
        for alpha in [0.7, 0.8, 0.9]:
            H_mix = alpha * best_H + (1 - alpha) * H_base
            H_mix = np.where(np.isfinite(H_mix) & (H_mix > 0), H_mix, 1e-9)
            
            costs = []
            for s in seeds:
                cost = _fast_aco_sim(dm, demands, capacity, H_mix, 
                                      n_ants=20, n_iterations=10, 
                                      seed=s, time_limit=5.0)
                if cost < np.inf:
                    costs.append(cost)
            
            if costs:
                avg_cost = np.mean(costs)
                if avg_cost < best_score:
                    best_score = avg_cost
                    best_H = H_mix

    # Final safety check
    best_H = np.where(np.isfinite(best_H) & (best_H > 0), best_H, 1e-9)
    
    return best_H

# EVOLVE-BLOCK-END
