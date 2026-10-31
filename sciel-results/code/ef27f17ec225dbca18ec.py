"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a Minimax heuristic.
    
    This approach minimizes the maximum of the distance from the current node
    to the candidate and the distance from the candidate to the destination.
    This balances the immediate movement cost with the future return cost,
    avoiding choices that are cheap now but expensive to resolve later.
    It is a robust strategy compared to additive weighted sums.
    """
    if len(unvisited_nodes) == 0:
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each unvisited node to the destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Minimax Score: Minimize the maximum of the two distances
    scores = np.maximum(dist_from_current, dist_to_dest)
    
    # Pick the node with the lowest score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
