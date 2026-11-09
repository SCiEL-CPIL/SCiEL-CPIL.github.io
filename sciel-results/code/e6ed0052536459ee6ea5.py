"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step look-ahead heuristic.
    Minimizes distance(current, candidate) + distance(candidate, nearest_other_unvisited).
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Distances from current node to all unvisited nodes
    dists_from_cur = distance_matrix[current_node, unvisited_nodes]
    
    # Get the submatrix of distances between unvisited nodes
    # Shape: (n_unvisited, n_unvisited)
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # We want the distance from each candidate to its nearest neighbor among the OTHER unvisited nodes.
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Find the minimum distance to any other unvisited node for each candidate
    min_dist_to_other = np.min(sub_matrix, axis=1)
    
    # Calculate combined score
    # We use a weight of 1.0 for the look-ahead term
    scores = dists_from_cur + min_dist_to_other
    
    # Select the candidate with the minimum score
    best_local_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_local_idx])
# EVOLVE-BLOCK-END
