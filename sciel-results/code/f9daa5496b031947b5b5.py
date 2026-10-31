"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Greedy Lookahead heuristic.
    
    This approach differs from the weighted-sum heuristic by explicitly simulating
    the next step in the tour. For each candidate node 'u', we estimate the cost of
    moving from 'current' to 'u' and then to the best possible next node 'v' (or
    destination if 'u' is the last unvisited node).
    
    Special handling is applied when only 2 unvisited nodes remain, where we can
    exactly evaluate the two possible orderings before closing the tour.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # Exact evaluation of the two possible paths:
        # Path 1: current -> u0 -> u1 -> destination
        # Path 2: current -> u1 -> u0 -> destination
        u0, u1 = unvisited_nodes[0], unvisited_nodes[1]
        
        cost_path1 = distance_matrix[current_node, u0] + distance_matrix[u0, u1] + distance_matrix[u1, destination_node]
        cost_path2 = distance_matrix[current_node, u1] + distance_matrix[u1, u0] + distance_matrix[u0, destination_node]
        
        if cost_path1 <= cost_path2:
            return int(u0)
        else:
            return int(u1)
    
    # For n_unvisited > 2, use 1-step lookahead
    # Score(u) = dist(current, u) + min_{v in unvisited \ {u}} dist(u, v)
    
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Compute min distance from each unvisited node to any other unvisited node
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(submatrix, np.inf)
    min_dist_to_other = np.min(submatrix, axis=1)
    
    scores = dist_from_current + min_dist_to_other
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
