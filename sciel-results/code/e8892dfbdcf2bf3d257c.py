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

    # Approach: Savings-Based Heuristic (Clarke-Wright inspired)
    # We prioritize nodes that provide high savings relative to the depot.
    # The standard savings for inserting j after i is S = d(0, i) + d(0, j) - d(i, j).
    # To maximize S, we minimize (d(i, j) - d(0, j)) since d(0, i) is constant.
    # This encourages visiting nodes that are close to the current position
    # relative to their distance from the depot, promoting clustered routes.
    
    dist_current_to_nodes = distance_matrix[current_node, feasible_nodes]
    dist_depot_to_nodes = distance_matrix[depot, feasible_nodes]
    
    # Minimize: distance from current - distance from depot
    scores = dist_current_to_nodes - dist_depot_to_nodes
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
