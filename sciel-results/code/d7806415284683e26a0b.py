import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen based on contract, but safe guard
        return destination_node 
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        # Exact calculation for the last two steps
        u1 = int(unvisited_nodes[0])
        u2 = int(unvisited_nodes[1])
        
        # Path: current -> u1 -> u2 -> destination
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        
        # Path: current -> u2 -> u1 -> destination
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost1 <= cost2:
            return u1
        else:
            return u2

    # General case: 2-step lookahead
    # For each candidate u, score = d(current, u) + min(d(u, v) for v in unvisited \ {u})
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    for i, u in enumerate(unvisited_nodes):
        # Distance from current to u
        d_cu = dist_from_current[i]
        
        # Find nearest neighbor v of u among unvisited nodes, excluding u
        # We can use the distance matrix row for u
        dist_from_u = distance_matrix[u, unvisited_nodes]
        
        # Mask out u itself
        # Since unvisited_nodes is an array of indices, we need to find the index of u in unvisited_nodes
        # i is the index of u in unvisited_nodes
        dist_from_u[i] = np.inf
        
        # If there's only one other node, min will pick it.
        # If there are more, it picks the closest.
        min_d_uv = np.min(dist_from_u)
        
        score = d_cu + min_d_uv
        
        if score < best_score:
            best_score = score
            best_node = u

    return int(best_node)
