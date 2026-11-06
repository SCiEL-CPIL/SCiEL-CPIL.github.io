"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: 120s limit, stop at 115s
    if time.time() - _start_time > 115:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    unvisited_arr = np.array(unvisited_nodes, dtype=int)
    
    # Filter feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # 1-step lookahead heuristic
    # Score = dist(current, node) + min(dist(node, depot), min(dist(node, other_feasible)))
    
    # Distances from current to feasible nodes
    dists_from_current = distance_matrix[current_node, feasible_nodes]
    
    # Distances from feasible nodes to depot
    dists_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Distances between feasible nodes
    # Create (N, N) matrix
    dists_between = distance_matrix[feasible_nodes[:, None], feasible_nodes[None, :]]
    
    # Set diagonal to inf to exclude self
    np.fill_diagonal(dists_between, np.inf)
    
    # Min distance to any other feasible node
    min_dist_to_other = np.min(dists_between, axis=1)
    
    # Min cost for next step
    min_next_cost = np.minimum(dists_to_depot, min_dist_to_other)
    
    # Total score
    scores = dists_from_current + min_next_cost
    
    best_idx = np.argmin(scores)
    
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
