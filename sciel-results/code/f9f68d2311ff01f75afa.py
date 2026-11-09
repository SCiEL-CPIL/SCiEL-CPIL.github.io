import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute heuristic matrix for ACO orienteering using a tour-aware approach.
    
    Structural Change:
    1. Tour-Aware Heuristic:
       Instead of using static distance-based heuristics, we compute a high-quality 
       greedy tour and reinforce edges that appear in this tour. This guides the ACO
       toward proven good paths rather than relying solely on local ratios.
       
    2. Hybrid Combination:
       We combine the tour reinforcement with a spatial density potential (KDE-like)
       that guides ants toward clusters of high-value targets.
       
    3. Feasibility Masking:
       Strict masking of infeasible moves (d[i,j] + d[j,0] > max_len).
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    eps = 1e-9
    
    # 1. Compute Total Cost: d[i,j] + d[j,0]
    total_cost = distance_matrix + distance_matrix[:, 0:1]
    
    # 2. Static Feasibility Mask
    feasible_mask = total_cost <= max_len
    
    # 3. Compute a high-quality greedy tour
    # Start at node 0, repeatedly pick the next node that maximizes prize/cost ratio
    # while staying within budget
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    tour_nodes = [0]
    current_length = 0.0
    
    for _ in range(n - 1):
        best_ratio = -1
        best_next = -1
        for j in range(n):
            if visited[j]:
                continue
            # Check if we can go to j and return to depot
            step_cost = distance_matrix[current, j]
            return_cost = distance_matrix[j, 0]
            if current_length + step_cost + return_cost > max_len:
                continue
            # Ratio: prize / step cost (prefer high prize, low cost)
            if distance_matrix[current, j] > eps:
                ratio = prizes[j] / distance_matrix[current, j]
            else:
                ratio = prizes[j] / eps
            if ratio > best_ratio:
                best_ratio = ratio
                best_next = j
        if best_next == -1:
            break
        current_length += distance_matrix[current, best_next]
        visited[best_next] = True
        current = best_next
        tour_nodes.append(best_next)
    
    # 4. Create tour edge reinforcement matrix
    # Edges in the tour get high attractiveness
    tour_edges = np.zeros((n, n), dtype=np.float64)
    for k in range(len(tour_nodes) - 1):
        i = tour_nodes[k]
        j = tour_nodes[k + 1]
        tour_edges[i, j] = 1.0
    
    # 5. Compute Spatial Density Potential (KDE-like)
    # For each node j, compute density based on nearby high-prize nodes
    # Using Gaussian kernel with bandwidth based on average distance
    avg_dist = np.mean(distance_matrix)
    bandwidth = max(avg_dist * 0.5, 0.05)
    
    # Density at node j = sum over k of prize[k] * exp(-d[j,k]^2 / (2*bandwidth^2))
    dist_sq = distance_matrix ** 2
    kernel = np.exp(-dist_sq / (2 * bandwidth ** 2))
    density = np.sum(kernel * prizes[None, :], axis=1)
    
    # Normalize density
    max_density = np.max(density)
    if max_density > eps:
        density_norm = density / max_density
    else:
        density_norm = np.zeros_like(density)
    
    # 6. Combine components
    # Base heuristic: prize[j] / d[i,j] (classic ACO heuristic)
    base_heuristic = prizes[None, :] / (distance_matrix + eps)
    
    # Tour reinforcement: boost edges in the greedy tour
    tour_boost = 1.0 + 2.0 * tour_edges
    
    # Density boost: guide toward high-density areas
    density_boost = 1.0 + 1.5 * density_norm[None, :]
    
    # Budget penalty: penalize moves that consume large fraction of budget
    budget_penalty = np.exp(-2.0 * total_cost / max_len)
    
    # Final heuristic
    eta = base_heuristic * tour_boost * density_boost * budget_penalty
    
    # 7. Apply Feasibility Mask
    eta = np.where(feasible_mask, eta, 1e-9)
    
    # Ensure node 0 is not a destination
    eta[:, 0] = 1e-9
    
    # 8. Normalization
    max_eta = np.max(eta)
    if max_eta > eps:
        eta = eta / max_eta
    else:
        eta = np.full_like(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Check deadline
    if time.time() > deadline:
        return eta
    
    return eta
