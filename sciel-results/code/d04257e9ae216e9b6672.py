"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step Lookahead Nearest Neighbor heuristic.
    
    This approach evaluates each unvisited node 'u' by summing the distance 
    from the current node to 'u' and the distance from 'u' to its nearest 
    other unvisited neighbor. This looks one step ahead to avoid getting 
    stuck in a dead end or a long backtrack. For the final two nodes, 
    it explicitly evaluates both orderings against the destination.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        u1 = int(unvisited_nodes[0])
        u2 = int(unvisited_nodes[1])
        
        # Cost of going u1 -> u2 -> destination
        cost_12 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        
        # Cost of going u2 -> u1 -> destination
        cost_21 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        
        if cost_12 < cost_21:
            return u1
        else:
            return u2

    # General case: n_unvisited > 2
    # We want to minimize d(current, u) + min(d(u, v) for v in unvisited, v != u)
    
    # Precompute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    best_score = np.inf
    best_node = int(unvisited_nodes[0])
    
    # Iterate through candidates
    for i, u in enumerate(unvisited_nodes):
        d1 = dists_from_current[i]
        
        # Find nearest neighbor of u among unvisited (excluding u itself)
        # We can use the distance matrix row for u, masked by unvisited
        # dists_u = distance_matrix[u, unvisited_nodes]
        # dists_u[i] is 0 (distance to self). We want min of others.
        
        # Efficient way: 
        # Get all distances from u to unvisited nodes
        dists_u_to_unvisited = distance_matrix[u, unvisited_nodes]
        # Mask out self (index i) by setting to inf
        dists_u_to_unvisited[i] = np.inf
        # The minimum distance to another unvisited node
        d2 = np.min(dists_u_to_unvisited)
        
        score = d1 + d2
        
        if score < best_score:
            best_score = score
            best_node = int(u)
            
    return best_node

# EVOLVE-BLOCK-END
