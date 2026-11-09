"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global time tracking to ensure we don't exceed the 150s limit
_start_time = time.time()
TIME_LIMIT = 148.0  # Safety margin before 150s

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Check time budget
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to pure nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # 1. Distance from current to each candidate
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from each candidate to destination
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Lookahead: Minimum distance from each candidate to any OTHER unvisited node
    # This helps avoid "dead ends" where a node is isolated from the rest of the cluster.
    # We extract the submatrix of distances between unvisited nodes.
    # Shape: (n_unvisited, n_unvisited)
    dist_matrix_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity so a node doesn't consider itself as the "next" node
    np.fill_diagonal(dist_matrix_unvisited, np.inf)
    
    # The minimum distance from each unvisited node to any other unvisited node
    min_dist_to_next = np.min(dist_matrix_unvisited, axis=1)
    
    # Handle case where a node might be the only one left in the subgraph logic (though n_unvisited > 1 here)
    # If min is inf, it means no other nodes, which shouldn't happen for n_unvisited > 1
    
    # Weighted combination:
    # w1: immediate cost
    # w2: destination proximity (to close the loop efficiently)
    # w3: connectivity to other unvisited nodes (to avoid stranding)
    w1 = 0.6
    w2 = 0.2
    w3 = 0.2
    
    # Normalize terms to similar scales? 
    # Distances are usually in similar ranges for uniform distribution, so direct weighting is okay.
    scores = w1 * dists_from_current + w2 * dists_to_dest + w3 * min_dist_to_next
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
