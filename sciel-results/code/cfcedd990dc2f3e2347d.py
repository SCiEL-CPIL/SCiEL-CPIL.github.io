"""Rewritten program for TSP constructive heuristic using constrained minimax strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a constrained minimax strategy:
    Prioritize nearest neighbors, but restrict the choice to nodes that have a
    'low bottleneck' (i.e., are not too far from the rest of the unvisited nodes).
    This prevents the tour from becoming fragmented or requiring long edges later.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    unvisited_idx = unvisited_nodes
    n_unvisited = len(unvisited_idx)
    
    # 1. Distance from current node to all unvisited nodes
    d_current = distance_matrix[current_node, unvisited_idx]
    
    # 2. Bottleneck cost: Max distance from each unvisited node to other unvisited nodes
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    # Copy to ensure we don't modify the original matrix if it happens to be a view
    submatrix_copy = submatrix.copy()
    np.fill_diagonal(submatrix_copy, -np.inf)
    bottleneck_costs = np.max(submatrix_copy, axis=1)
    
    # 3. Constrained Minimax Strategy
    # We restrict candidates to those with a bottleneck cost below the 25th percentile.
    # This forces the tour to move towards the "center" of the remaining unvisited nodes,
    # reducing the risk of long edges needed to connect distant clusters later.
    
    # If there are very few nodes, the percentile might be too restrictive or noisy.
    # For n_unvisited <= 4, we just use nearest neighbor to avoid over-constraining.
    if n_unvisited <= 4:
        best_idx = int(np.argmin(d_current))
    else:
        threshold = np.percentile(bottleneck_costs, 25)
        good_mask = bottleneck_costs <= threshold
        
        # If for some reason no node passes the threshold (should not happen with percentile),
        # fallback to nearest.
        if np.sum(good_mask) == 0:
            best_idx = int(np.argmin(d_current))
        else:
            # Among the "good" (central) candidates, pick the nearest one.
            # Set scores of bad candidates to infinity.
            scores = np.where(good_mask, d_current, np.inf)
            best_idx = int(np.argmin(scores))
            
    return int(unvisited_idx[best_idx])
# EVOLVE-BLOCK-END
