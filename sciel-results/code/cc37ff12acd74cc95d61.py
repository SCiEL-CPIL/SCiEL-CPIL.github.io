import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    unvisited_idx = unvisited_nodes
    
    # Distance from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_idx]
    
    # Calculate bottleneck cost: max distance from each unvisited node to any other unvisited node
    # submatrix is NxN where N is number of unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    np.fill_diagonal(submatrix, -np.inf)
    bottleneck_costs = np.max(submatrix, axis=1)
    
    # Minimax strategy: Minimize the maximum of (distance from current, bottleneck to others)
    # This ensures we don't take a very long step, and we don't pick a node that is an outlier relative to the rest
    scores = np.maximum(dist_from_current, bottleneck_costs)
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_idx[best_idx])
