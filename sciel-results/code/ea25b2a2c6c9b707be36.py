"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node to visit using a greedy heuristic with 2-step lookahead.
    Instead of just picking the nearest neighbor, it estimates the cost of moving
    to the candidate and then to the candidate's nearest neighbor among the remaining
    unvisited nodes. This helps avoid "dead ends" where a node is close to the current
    position but far from the rest of the tour.
    """
    n_unvisited = len(unvisited_nodes)
    
    # Base cases
    if n_unvisited == 0:
        # Should not happen in a valid TSP construction loop, but return -1 as a fallback
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Extract distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Extract the sub-matrix of distances between unvisited nodes
    # This is a K x K matrix where K is the number of unvisited nodes
    # Using np.ix_ for efficient fancy indexing
    unvisited_indices = unvisited_nodes
    D_unvisited = distance_matrix[np.ix_(unvisited_indices, unvisited_indices)]
    
    # For each unvisited node u (row i), we want the distance to its nearest neighbor
    # among the *other* unvisited nodes.
    # We achieve this by setting the diagonal to infinity and taking the row minimum.
    np.fill_diagonal(D_unvisited, np.inf)
    
    # min_dist_to_other[i] is the distance from unvisited_nodes[i] to its closest neighbor
    # in the unvisited set (excluding itself).
    min_dist_to_other = np.min(D_unvisited, axis=1)
    
    # The score for each candidate is the immediate cost plus the estimated next cost.
    # This is a 2-step lookahead greedy choice.
    scores = dists_from_current + min_dist_to_other
    
    # Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
