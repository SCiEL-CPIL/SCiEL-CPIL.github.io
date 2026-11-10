"""Optimized heuristics for ACO CVRP with remaining-capacity-potential field."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using a remaining-capacity-potential field.
    
    Strategy:
    1. Precompute a greedy nearest-neighbor tour from the depot to estimate the
       visit order and residual vehicle capacity at each node's position.
    2. For each edge i->j, modulate attractiveness by the ratio of destination
       demand to estimated residual capacity at the source node's position.
       This strongly penalizes edges that risk exceeding the vehicle limit.
    3. Combine with normalized inverse distance for spatial guidance.
    
    The capacity-potential field encourages early, compact route formation by
    making edges to high-demand nodes less attractive when the estimated
    remaining capacity is low.
    """
    deadline = time.time() + 290.0
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n == 0:
        return np.array([])
        
    if d.shape != (n, n):
        return 1.0 / (d + 1e-9)
        
    eps = 1e-9
    demands = np.asarray(demands, dtype=np.float64)
    
    # Step 1: Precompute a greedy nearest-neighbor tour to estimate visit positions
    # and residual capacity at each node.
    # Start at depot (node 0), repeatedly visit the nearest unvisited customer
    # that fits in remaining capacity.
    
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    tour = [0]
    node_position = np.full(n, -1, dtype=np.int32)
    residual_cap_at_node = np.full(n, capacity, dtype=np.float64)
    
    for pos in range(1, n):
        # Find nearest unvisited customer that fits
        best_next = -1
        best_dist = np.inf
        for j in range(1, n):
            if not visited[j] and demands[j] <= remaining_cap:
                dist = d[current, j]
                if dist < best_dist:
                    best_dist = dist
                    best_next = j
        
        if best_next == -1:
            # No customer fits; break (shouldn't happen with capacity 50 and max demand 9)
            break
        
        visited[best_next] = True
        tour.append(best_next)
        node_position[best_next] = pos
        remaining_cap -= demands[best_next]
        residual_cap_at_node[best_next] = remaining_cap
        current = best_next
    
    # For any unvisited nodes (should be none), assign default position and residual
    for j in range(1, n):
        if not visited[j]:
            node_position[j] = n - 1
            residual_cap_at_node[j] = 0.0
    
    # Step 2: Compute the capacity-potential field
    # For edge i->j, the potential is based on:
    # - demand[j] / max(residual_cap_at_node[i], eps)
    # Low residual capacity at source i makes high-demand destinations j less attractive.
    # We use a penalty factor: penalty[i,j] = 1 / (1 + alpha * demand[j] / residual[i])
    # where residual[i] is the estimated remaining capacity when at node i.
    
    # For the depot (i=0), residual capacity is full (capacity)
    residual_at_source = np.full(n, capacity, dtype=np.float64)
    for j in range(1, n):
        if node_position[j] >= 0:
            residual_at_source[j] = residual_cap_at_node[j]
    
    # Ensure no zero or negative residual
    residual_at_source = np.maximum(residual_at_source, eps)
    
    # demand[j] / residual_at_source[i] for all i,j
    demand_ratio = np.outer(np.ones(n), demands) / residual_at_source[:, np.newaxis]
    
    # Penalty: 1 / (1 + alpha * demand_ratio)
    # alpha controls the strength of capacity-awareness
    alpha = 2.0
    capacity_penalty = 1.0 / (1.0 + alpha * demand_ratio)
    
    # Normalize the penalty to have mean 1.0 off-diagonal
    mask = ~np.eye(n, dtype=bool)
    mean_penalty = np.mean(capacity_penalty[mask])
    if mean_penalty > 0 and np.isfinite(mean_penalty):
        capacity_penalty = capacity_penalty / mean_penalty
    else:
        capacity_penalty = np.ones((n, n))
    
    # Step 3: Base heuristic: Normalized inverse distance
    eta_base = 1.0 / ((d + eps) ** 1.5)
    mean_eta = np.mean(eta_base[mask])
    if mean_eta > 0 and np.isfinite(mean_eta):
        eta_base = eta_base / mean_eta
    else:
        eta_base = 1.0 / ((d + eps) ** 1.5)
    
    # Step 4: Combine base heuristic with capacity penalty
    # Multiplicative combination: eta = eta_base * capacity_penalty
    eta = eta_base * capacity_penalty
    
    # Re-normalize
    mean_final = np.mean(eta[mask])
    if mean_final > 0 and np.isfinite(mean_final):
        eta = eta / mean_final
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
