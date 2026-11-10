"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    if len(unvisited_nodes) == 0:
        return depot

    # Filter to feasible nodes (demand fits in remaining capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    # Approach: Greedy Lookahead (2-step)
    # Score each candidate u by d(current, u) + min(d(u, v) for v in feasible \ {u}, d(u, depot))
    # This approximates the cost of the next two edges in the route.
    
    scores = np.full(len(feasible_nodes), np.inf)
    d_to_depot = distance_matrix[feasible_nodes, depot]
    
    for i, u in enumerate(feasible_nodes):
        d_cu = distance_matrix[current_node, u]
        
        # Find the minimum distance from u to any other feasible node
        # We exclude u itself to ensure we look at a *next* node
        # If there is only one feasible node, the next step must be depot
        if len(feasible_nodes) > 1:
            # Get distances from u to all other feasible nodes
            d_u_to_others = distance_matrix[u, feasible_nodes]
            d_u_to_others[i] = np.inf # Exclude self
            min_next_dist = np.min(d_u_to_others)
        else:
            min_next_dist = np.inf
            
        # The next step from u is either to another customer or back to depot
        next_step_cost = min(min_next_dist, d_to_depot[i])
        
        scores[i] = d_cu + next_step_cost

    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
