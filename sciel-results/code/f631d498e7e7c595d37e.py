"""Rewritten program for TSP constructive heuristic using minimax strategy."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a minimax strategy:
    Minimizes the maximum of the immediate edge cost and the bottleneck
    distance from the candidate to the rest of the unvisited set.
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
    # We calculate the max distance from each candidate to any other unvisited node.
    submatrix = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)].copy()
    # The diagonal is 0. Since n_unvisited > 1, the max of the row will be
    # the maximum distance to any other node in the set.
    bottleneck_costs = np.max(submatrix, axis=1)
    
    # 3. Minimax Score: Max(immediate_cost, bottleneck_cost)
    # This ensures we don't pick a node that is very far away (high immediate cost)
    # OR a node that is very isolated from the rest (high bottleneck cost).
    scores = np.maximum(d_current, bottleneck_costs)
    
    # 4. Penalize visiting the destination node early if it's in unvisited
    # This ensures we return to the start/destination only at the end
    if destination_node in unvisited_idx:
        dest_mask = (unvisited_idx == destination_node)
        if np.sum(dest_mask) > 0:
            # If there are other nodes to visit, penalize the destination
            if n_unvisited > 1:
                scores[dest_mask] += 1000.0
    
    # 5. Select the node with the minimum score
    best_idx = int(np.argmin(scores))
    
    return int(unvisited_idx[best_idx])
# EVOLVE-BLOCK-END
