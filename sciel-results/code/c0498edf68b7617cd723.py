"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    Minimizes d(current, j) + min(d(j, k) for k in feasible_unvisited \ {j}).
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Ensure unvisited_nodes is an array for indexing
    unvisited_arr = np.array(unvisited_nodes)
    
    # Filter nodes that fit in the remaining capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Calculate distance from current node to all feasible candidates
    dist_current = distance_matrix[current_node, feasible_nodes]

    # Calculate the minimum distance from each candidate to any other feasible candidate
    # This creates a submatrix of distances between all pairs of feasible nodes
    # Using np.ix_ for efficient fancy indexing
    sub_matrix = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # Exclude self-loops by setting diagonal to infinity
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Find the closest neighbor for each candidate
    min_next_dist = np.min(sub_matrix, axis=1)

    # Total score: cost to get to candidate + estimated cost to get to next candidate
    scores = dist_current + min_next_dist

    # Select the candidate with the lowest score
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
