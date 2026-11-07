"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2

def _fast_aco_sim(dm, demands, capacity, H, n_ants=20, n_iterations=10, seed=42, time_limit=5.0):
    """
    Faster, higher-fidelity simulation of the evaluator's ACO.
    Matches n_ants=20.
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


def _compute_cluster_heuristic(dm, coords, demands, capacity, n_clusters, gamma, cross_penalty=0.1):
    """
    Compute a heuristic matrix based on K-Means clustering.
    Edges within the same cluster get full weight 1/d^gamma.
    Edges between clusters get weight cross_penalty/d^gamma.
    """
    n = dm.shape[0]
    
    # Base distance heuristic
    with np.errstate(divide='ignore', invalid='ignore'):
        dm_safe = np.where(dm > 1e-10, dm, 1e10)
        base_eta = 1.0 / np.power(dm_safe, gamma)
    
    # Ensure positive and finite
    base_eta[~np.isfinite(base_eta)] = 1e-9
    base_eta[base_eta <= 0] = 1e-9
    
    # If n_clusters is 1 or invalid, return base heuristic
    if n_clusters <= 1 or n_clusters >= n:
        np.fill_diagonal(base_eta, 1e-9)
        return base_eta

    # Perform K-Means clustering on customers (nodes 1 to n-1)
    customer_coords = coords[1:]
    
    try:
        actual_k = min(n_clusters, n - 1)
        if actual_k <= 1:
            np.fill_diagonal(base_eta, 1e-9)
            return base_eta
            
        # Use a fixed seed for reproducibility in clustering
        centroids, labels = kmeans2(customer_coords, actual_k, minit='points', seed=42)
        
        # Assign depot (0) to the cluster of the nearest centroid
        dists_to_centroids = np.linalg.norm(centroids - coords[0], axis=1)
        depot_cluster = np.argmin(dists_to_centroids)
        
        # Create full label array
        full_labels = np.zeros(n, dtype=int)
        full_labels[0] = depot_cluster
        full_labels[1:] = labels
        
        # Apply penalty
        heuristic = base_eta.copy()
        # Vectorized penalty: create a boolean matrix of cross-cluster edges
        cross_mask = (full_labels[:, None] != full_labels[None, :])
        heuristic[cross_mask] *= cross_penalty
        
        np.fill_diagonal(heuristic, 1e-9)
        return heuristic
        
    except Exception:
        # Fallback to base heuristic if clustering fails
        np.fill_diagonal(base_eta, 1e-9)
        return base_eta


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Returns an edge-attractiveness matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    
    # Estimate minimum number of vehicles
    total_demand = np.sum(demands[1:])
    min_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    
    # Define search space for structural parameters
    # K: number of clusters
    # Gamma: distance exponent
    k_values = [min_vehicles, min_vehicles + 1, min_vehicles + 2]
    gamma_values = [1.0, 1.5, 2.0, 2.5]
    
    candidates = []
    
    # Generate cluster-based heuristics
    for k in k_values:
        for gamma in gamma_values:
            if time.time() > deadline - 30.0:
                break
            h = _compute_cluster_heuristic(dm, coords, demands, capacity, k, gamma, cross_penalty=0.1)
            h_safe = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
            candidates.append(h_safe)
    
    # Also add simple distance-based heuristics as fallbacks
    H1 = 1.0 / dm_safe
    H2 = 1.0 / (dm_safe ** 2)
    h1_safe = np.where(np.isfinite(H1) & (H1 > 0), H1, 1e-9)
    h2_safe = np.where(np.isfinite(H2) & (H2 > 0), H2, 1e-9)
    candidates.append(h1_safe)
    candidates.append(h2_safe)
    
    best_H = candidates[0]
    best_score = np.inf
    
    # Evaluate candidates
    for i, h in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
            
        costs = []
        for s in [0, 1, 2]:
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

    # Refine with mixing
    if time.time() < deadline - 40.0:
        for alpha in [0.5, 0.7, 0.8, 0.9]:
            H_mix = alpha * best_H + (1 - alpha) * h1_safe
            H_mix = np.where(np.isfinite(H_mix) & (H_mix > 0), H_mix, 1e-9)
            
            costs = []
            for s in [0, 1, 2]:
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

    return best_H

# EVOLVE-BLOCK-END
