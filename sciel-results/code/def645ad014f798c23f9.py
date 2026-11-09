import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1 # Should not happen based on contract, but safe
        
    # Get distances from current to all unvisited
    dists = distance_matrix[current_node, unvisited_nodes]
    
    # If only 1 node left, return it
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Select top K candidates
    # K should be small, e.g., 5
    k = min(5, n_unvisited)
    # Get indices of the k smallest distances
    # argsort gives indices in sorted order
    top_k_indices = np.argsort(dists)[:k]
    
    best_cost = float('inf')
    best_node = unvisited_nodes[top_k_indices[0]]
    
    # Precompute distances from destination to all unvisited? 
    # We need d(c, destination) for the last step case.
    # And d(c, nn) for the lookahead.
    
    for idx in top_k_indices:
        candidate = unvisited_nodes[idx]
        
        # Cost to reach candidate
        cost_to_candidate = dists[idx]
        
        # Determine the next step cost
        if n_unvisited == 1:
            # This case is handled above, but logic holds
            next_cost = distance_matrix[candidate, destination_node]
        else:
            # Find nearest neighbor of candidate among the OTHER unvisited nodes
            # We need to exclude 'candidate' from the unvisited set for this calculation
            # Since unvisited_nodes is an array, we can create a mask or just check all and skip candidate
            
            # Efficient way: 
            # Get distances from candidate to all unvisited
            dists_from_candidate = distance_matrix[candidate, unvisited_nodes]
            
            # Set the distance to the candidate itself to infinity to exclude it
            # The index of candidate in unvisited_nodes is idx
            dists_from_candidate[idx] = np.inf
            
            # The nearest neighbor is the argmin
            nn_idx = np.argmin(dists_from_candidate)
            next_cost = dists_from_candidate[nn_idx]
            
        total_cost = cost_to_candidate + next_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = candidate
            
    return int(best_node)
