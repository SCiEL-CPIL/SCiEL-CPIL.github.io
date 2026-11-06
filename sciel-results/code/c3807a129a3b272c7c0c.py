# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    if n_unvisited == 2:
        u1 = unvisited_nodes[0]
        u2 = unvisited_nodes[1]
        
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost1 < cost2:
            return int(u1)
        else:
            return int(u2)

    # General case: 1-step lookahead
    # Score(u) = dist(current, u) + min(dist(u, v) for v in unvisited, v != u)
    # We can vectorize this partially.
    
    # dist_current_to_all = distance_matrix[current_node, unvisited_nodes]
    # For each u_i, we need min(dist(u_i, u_j) for j != i)
    
    # Let's use a loop, it's fast enough for N=50
    scores = np.empty(n_unvisited)
    for i in range(n_unvisited):
        u = unvisited_nodes[i]
        d_cu = distance_matrix[current_node, u]
        
        # Distances from u to all other unvisited
        d_u_to_others = distance_matrix[u, unvisited_nodes]
        d_u_to_others[i] = np.inf
        
        min_next = np.min(d_u_to_others)
        
        scores[i] = d_cu + min_next
        
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
