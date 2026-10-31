"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using an adaptive nearest-neighbor heuristic with dynamic
    weighting based on the number of remaining unvisited nodes.
    
    When few nodes remain, we prioritize getting to the destination efficiently.
    When many nodes remain, we use a more balanced approach to avoid local minima.
    
    The score for each candidate node u is:
        score(u) = dist(current, u) + alpha(n) * dist(u, destination) + beta(n) * min_dist(u, unvisited_others)
    
    where alpha and beta are functions of the number of remaining unvisited nodes.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return int(destination_node)
    
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited node to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute isolation penalty: min distance from each unvisited node to any other unvisited node
    submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    np.fill_diagonal(submatrix, np.inf)
    min_dist_to_other = np.min(submatrix, axis=1)
    
    # Adaptive weights based on remaining nodes
    # When n is small, increase weight on destination to close the tour efficiently
    # When n is large, use more balanced weights
    if n <= 5:
        alpha = 1.0
        beta = 0.1
    elif n <= 10:
        alpha = 0.7
        beta = 0.2
    elif n <= 20:
        alpha = 0.5
        beta = 0.3
    else:
        alpha = 0.4
        beta = 0.4
    
    # Compute the combined score
    scores = dist_from_current + alpha * dist_to_dest + beta * min_dist_to_other
    
    # Pick the node with the lowest score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
