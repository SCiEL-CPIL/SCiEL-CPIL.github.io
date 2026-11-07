"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def _fast_aco_sim(dm, demands, capacity, H, n_ants=20, n_iterations=30, seed=42, time_limit=5.0):
    """
    Fast simulation of the evaluator's ACO with full 30 iterations.
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


def _generate_heuristic_candidates(dm, coords, demands, capacity, n):
    """
    Generate a diverse set of heuristic candidates based on different principles.
    """
    candidates = []
    
    # 1. Classic inverse distance
    dm_safe = np.where(dm > 1e-10, dm, 1e10)
    H1 = 1.0 / dm_safe
    candidates.append(("inv_dist", H1))
    
    # 2. Inverse squared distance
    H2 = 1.0 / (dm_safe ** 2)
    candidates.append(("inv_sq_dist", H2))
    
    # 3. Demand-aware: penalize edges to high-demand customers
    d_demands = demands.reshape(-1, 1)
    H3 = 1.0 / (dm_safe * (1.0 + 0.15 * d_demands))
    candidates.append(("demand_penalty", H3))
    
    # 4. Depot-centric: boost edges to customers near the depot
    dist_depot = dm[0, :]
    dist_depot_safe = np.where(dist_depot > 1e-10, dist_depot, 1e10)
    boost = 1.0 / (1.0 + 0.5 * dist_depot_safe).T
    H4 = H1 * boost
    candidates.append(("depot_boost", H4))
    
    # 5. Cluster-aware: use k-means-like clustering to create regional preferences
    # Simple approach: compute mean distance to each customer's neighborhood
    # For speed, use a simplified version: average distance to k nearest neighbors
    k = 5
    # For each node, compute mean distance to its k nearest neighbors (excluding self)
    neighbor_dists = np.zeros(n)
    for i in range(n):
        d = dm[i, :].copy()
        d[i] = np.inf
        nearest = np.argpartition(d, k)[:k]
        neighbor_dists[i] = np.mean(d[nearest])
    
    # Normalize
    neighbor_dists_safe = np.where(neighbor_dists > 1e-10, neighbor_dists, 1e10)
    # Edges to well-connected nodes (low neighbor_dists) are more attractive
    H5 = H1 * (1.0 / (1.0 + 0.3 * neighbor_dists_safe.T))
    candidates.append(("cluster_aware", H5))
    
    # 6. Angle-based: prefer edges that maintain direction (less backtracking)
    # Compute angle between consecutive edges
    # For each pair (i,j), check if the direction from depot to i and i to j are similar
    # Simplified: use dot product of vectors
    coords = np.asarray(coords, dtype=np.float64)
    depot = coords[0]
    
    H6 = np.ones_like(dm)
    for i in range(n):
        for j in range(n):
            if i == j or i == 0 or j == 0:
                continue
            vec1 = coords[i] - depot
            vec2 = coords[j] - coords[i]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 1e-10 and norm2 > 1e-10:
                cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                # Higher cosine = more aligned direction
                H6[i, j] = 1.0 / dm_safe[i, j] * (1.0 + 0.5 * max(0, cos_angle))
            else:
                H6[i, j] = 1.0 / dm_safe[i, j]
    
    candidates.append(("angle_aware", H6))
    
    # 7. Load-balancing: prefer edges that keep remaining capacity balanced
    # Estimate: for each edge i->j, the "cost" in terms of capacity usage
    H7 = np.ones_like(dm)
    for i in range(n):
        for j in range(n):
            if i == j or i == 0:
                continue
            # Prefer edges to customers with moderate demand (not too high, not too low)
            demand_factor = 1.0 / (1.0 + abs(demands[j] - capacity/10.0) / (capacity/10.0))
            H7[i, j] = 1.0 / dm_safe[i, j] * demand_factor
    candidates.append(("load_balance", H7))
    
    # 8. Hybrid: combine inverse distance with demand and depot effects
    H8 = H1 * (1.0 + 0.1 * (1.0 - demands.reshape(-1, 1)/capacity)) * boost
    candidates.append(("hybrid", H8))
    
    # Ensure all candidates are valid
    valid_candidates = []
    for name, h in candidates:
        h_safe = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        valid_candidates.append((name, h_safe))
    
    return valid_candidates


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO.
    Returns an edge-attractiveness matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 310.0  # Safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    
    # Generate diverse candidates
    candidates = _generate_heuristic_candidates(dm, coords, demands, capacity, n)
    
    best_H = candidates[0][1]
    best_score = np.inf
    best_name = candidates[0][0]
    
    # Phase 1: Quick evaluation of all candidates with 1-2 seeds
    time_per_candidate = 15.0
    for name, h in candidates:
        if time.time() > deadline - 60.0:
            break
        
        costs = []
        for s in [0, 1]:
            cost = _fast_aco_sim(dm, demands, capacity, h, 
                                n_ants=20, n_iterations=30, 
                                seed=s, time_limit=5.0)
            if cost < np.inf:
                costs.append(cost)
        
        if costs:
            avg_cost = np.mean(costs)
            if avg_cost < best_score:
                best_score = avg_cost
                best_H = h
                best_name = name
    
    # Phase 2: Refine the top 3 candidates with more seeds
    if time.time() < deadline - 40.0:
        # Re-evaluate top candidates with more seeds
        refined_candidates = []
        for name, h in candidates:
            if time.time() > deadline - 30.0:
                break
            costs = []
            for s in [0, 1, 2, 3]:
                cost = _fast_aco_sim(dm, demands, capacity, h, 
                                    n_ants=20, n_iterations=30, 
                                    seed=s, time_limit=5.0)
                if cost < np.inf:
                    costs.append(cost)
            
            if costs:
                avg_cost = np.mean(costs)
                refined_candidates.append((name, h, avg_cost))
        
        if refined_candidates:
            refined_candidates.sort(key=lambda x: x[2])
            best_H = refined_candidates[0][1]
            best_score = refined_candidates[0][2]
            best_name = refined_candidates[0][0]
    
    # Phase 3: Try mixtures of top candidates
    if time.time() < deadline - 20.0:
        # Get top 2 candidates
        sorted_candidates = sorted([(name, h) for name, h in candidates], 
                                   key=lambda x: np.mean(x[1]))
        
        if len(sorted_candidates) >= 2:
            top1_name, top1_H = sorted_candidates[0]
            top2_name, top2_H = sorted_candidates[1]
            
            for alpha in [0.3, 0.5, 0.7]:
                if time.time() > deadline - 10.0:
                    break
                
                H_mix = alpha * top1_H + (1 - alpha) * top2_H
                H_mix = np.where(np.isfinite(H_mix) & (H_mix > 0), H_mix, 1e-9)
                
                costs = []
                for s in [0, 1, 2]:
                    cost = _fast_aco_sim(dm, demands, capacity, H_mix, 
                                        n_ants=20, n_iterations=30, 
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
