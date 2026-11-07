"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _fast_aco_sim_v4(dm, demands, capacity, H, n_ants=20, n_iterations=10, seed=42, time_limit=5.0):
    """
    Fast, vectorized-friendly simulation of the evaluator's ACO.
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
    
    candidates = [H_base]

    # 2. MST-based Heuristic (Structural Change)
    # Compute Minimum Spanning Tree using Prim's algorithm (vectorized)
    # This encourages ants to follow the global skeleton of the optimal tour.
    try:
        # Prim's algorithm
        in_mst = np.zeros(n, dtype=bool)
        in_mst[0] = True
        min_edge = dm[0, :].copy()
        min_edge[0] = np.inf
        
        for _ in range(n - 1):
            # Find the node with the minimum edge weight not in MST
            # We need to find min of min_edge where in_mst is False
            # To do this efficiently, we can just loop since n is small (50)
            # Or use numpy: np.argmin(np.where(in_mst, np.inf, min_edge))
            u = np.argmin(np.where(in_mst, np.inf, min_edge))
            
            in_mst[u] = True
            # Update min_edge for neighbors
            # min_edge = np.minimum(min_edge, dm[u, :])
            # But we only care about nodes not in MST for the next step
            # Actually, updating all is fine, argmin will ignore in_mst
            new_edges = dm[u, :]
            min_edge = np.minimum(min_edge, new_edges)
            min_edge[u] = np.inf # Prevent picking self again if not handled by mask

        # Build MST adjacency matrix (binary)
        # We need to reconstruct which edges were picked.
        # The above loop doesn't store the parent. Let's redo or store parents.
        
        # Redo Prim with parent tracking
        in_mst = np.zeros(n, dtype=bool)
        in_mst[0] = True
        min_edge = dm[0, :].copy()
        min_edge[0] = np.inf
        parent = np.zeros(n, dtype=int)
        
        for _ in range(n - 1):
            u = np.argmin(np.where(in_mst, np.inf, min_edge))
            in_mst[u] = True
            # parent[u] is already the node that led to u? 
            # No, min_edge[u] is the weight. We need to track which node gave that weight.
            # Let's track parent properly.
            
            # Actually, simpler:
            # min_edge[v] is the weight of the edge (u, v) where u is in MST.
            # We need to know which u.
            # Let's use a 2D array or just re-run with parent tracking.
            
            # Let's just re-run cleanly.
            pass
        
        # Clean Prim implementation
        in_mst = np.zeros(n, dtype=bool)
        in_mst[0] = True
        # min_edge[v] = weight of edge (u, v) connecting v to MST
        min_edge = dm[0, :].copy()
        min_edge[0] = np.inf
        # parent[v] = u, the node in MST connected to v
        parent = np.zeros(n, dtype=int)
        
        for _ in range(n - 1):
            # Find u with min weight not in MST
            u = np.argmin(np.where(in_mst, np.inf, min_edge))
            in_mst[u] = True
            
            # Update min_edge and parent for neighbors of u
            for v in range(n):
                if not in_mst[v] and dm[u, v] < min_edge[v]:
                    min_edge[v] = dm[u, v]
                    parent[v] = u

        # Build MST mask
        mst_mask = np.zeros((n, n), dtype=bool)
        for v in range(1, n):
            u = parent[v]
            mst_mask[u, v] = True
            mst_mask[v, u] = True

        # Heuristic: High attractiveness for MST edges, low for others
        # We want to strongly encourage MST edges.
        # H_mst = H_base * (10.0 if mst else 0.1)
        # Or additive: H_mst = H_base + bonus
        # Let's try multiplicative with strong contrast.
        H_mst = H_base * np.where(mst_mask, 10.0, 0.1)
        H_mst = np.where(np.isfinite(H_mst) & (H_mst > 0), H_mst, 1e-9)
        candidates.append(H_mst)

        # Also try a blend
        H_mst_blend = 0.5 * H_base + 0.5 * H_mst
        H_mst_blend = np.where(np.isfinite(H_mst_blend) & (H_mst_blend > 0), H_mst_blend, 1e-9)
        candidates.append(H_mst_blend)

    except Exception:
        pass

    # 3. K-Means Cluster-Based Heuristic
    H_cluster = H_base.copy()
    try:
        from scipy.cluster.vq import kmeans2
        
        k = max(2, int(np.sqrt(n)))
        coords_centered = coords - np.mean(coords, axis=0)
        coords_scaled = coords_centered / np.std(coords_centered, axis=0)
        
        centroids, labels = kmeans2(coords_scaled, k, minit='++', seed=42)
        
        same_cluster = (labels[:, None] == labels[None, :])
        H_cluster = H_base * np.where(same_cluster, 3.0, 0.2)
        H_cluster = np.where(np.isfinite(H_cluster) & (H_cluster > 0), H_cluster, 1e-9)
        candidates.append(H_cluster)
        
    except Exception:
        pass

    # 4. Nearest Neighbor Heuristic
    H_knn = np.zeros_like(dm)
    try:
        tree = cKDTree(coords)
        knn_dists, knn_idxs = tree.query(coords, k=6)
        
        for i in range(n):
            for rank in range(1, 6):
                j = knn_idxs[i, rank]
                d = knn_dists[i, rank]
                weight = (1.0 / (rank + 1)) * (1.0 / (1.0 + d))
                H_knn[i, j] = weight
                H_knn[j, i] = weight
        H_knn = np.where(np.isfinite(H_knn) & (H_knn > 0), H_knn, 1e-9)
        candidates.append(H_knn)
    except Exception:
        pass

    # 5. Demand-weighted Heuristic
    d_demands = demands.reshape(-1, 1)
    H_dem = 1.0 / (dm_safe * (1.0 + 0.1 * d_demands))
    H_dem = np.where(np.isfinite(H_dem) & (H_dem > 0), H_dem, 1e-9)
    candidates.append(H_dem)

    # Sanitize candidates
    sanitized_candidates = []
    for h in candidates:
        h_safe = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        sanitized_candidates.append(h_safe)
        
    best_H = sanitized_candidates[0]
    best_score = np.inf
    
    # Evaluate candidates using fast ACO sim
    seeds = [0, 1, 2]
    
    for i, h in enumerate(sanitized_candidates):
        if time.time() > deadline - 20.0:
            break
            
        costs = []
        for s in seeds:
            cost = _fast_aco_sim_v4(dm, demands, capacity, h, 
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
        for alpha in [0.7, 0.8, 0.9]:
            H_mix = alpha * best_H + (1 - alpha) * H_base
            H_mix = np.where(np.isfinite(H_mix) & (H_mix > 0), H_mix, 1e-9)
            
            costs = []
            for s in seeds:
                cost = _fast_aco_sim_v4(dm, demands, capacity, H_mix, 
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
