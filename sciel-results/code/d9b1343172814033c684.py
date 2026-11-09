import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in the TSP tour using a lookahead heuristic.
    
    Parameters:
    current_node (int): The node we are currently at.
    destination_node (int): The node we must return to (start node).
    unvisited_nodes (np.ndarray): Array of unvisited node indices.
    distance_matrix (np.ndarray): NxN matrix of pairwise distances.
    
    Returns:
    int: The index of the next node to visit.
    """
    # If no unvisited nodes, this function shouldn't be called, but handle gracefully.
    if len(unvisited_nodes) == 0:
        return -1 # Or handle based on contract, but contract implies it's called when unvisited exist.

    # If only one unvisited node, must pick it.
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Get distances from current node to all unvisited nodes
    # unvisited_nodes is an array of indices.
    # distance_matrix[current_node, unvisited_nodes] gives distances from current to each unvisited.
    
    # We need to evaluate each candidate c in unvisited_nodes.
    # Score(c) = d(current, c) + min_{c' in unvisited, c' != c} ( d(c, c') + d(c', destination) )
    
    # Optimization: Precompute d(c', destination) for all c' in unvisited.
    dest_costs = distance_matrix[unvisited_nodes, destination_node]
    
    # We need the min of (d(c, c') + dest_cost[c']) for c' != c.
    # This is slightly tricky to vectorize perfectly without N^2 loops, but N=50 is small.
    # We can just loop over candidates.
    
    best_score = float('inf')
    best_node = int(unvisited_nodes[0])
    
    # To speed up, we can vectorize the inner min if we treat it carefully, 
    # but a simple loop over ~50 candidates is fine.
    # d(c, unvisited) is a vector of length K.
    # We need min(d(c, c') + dest_cost[c']) excluding c.
    
    # Let's iterate
    for i, c in enumerate(unvisited_nodes):
        # Cost to get here
        cost_to_c = distance_matrix[current_node, c]
        
        # Cost from c to any other unvisited c', plus c' to destination
        # dists_from_c = distance_matrix[c, unvisited_nodes]
        # future_costs = dists_from_c + dest_costs
        # We need min(future_costs) excluding index i.
        
        dists_from_c = distance_matrix[c, unvisited_nodes]
        future_costs = dists_from_c + dest_costs
        
        # Mask out self
        # future_costs[i] = inf
        # min_val = np.min(future_costs)
        
        # Efficient min excluding one element
        # If K is small, just set to inf and min
        # But creating a copy every time is slow?
        # Alternative: 
        # min1, min2 approach?
        
        # Given N=50, K<=50.
        # Simple approach:
        # If K > 1:
        #   mask = np.ones(K, dtype=bool)
        #   mask[i] = False
        #   min_future = np.min(future_costs[mask])
        
        # Even simpler for small K:
        if len(unvisited_nodes) > 1:
             # Create a copy to avoid modifying original? 
             # Or use np.where?
             # np.min(np.where(np.arange(len(unvisited_nodes)) == i, np.inf, future_costs))
             min_future = np.min(np.where(np.arange(len(unvisited_nodes)) == i, np.inf, future_costs))
        else:
             min_future = 0 # Should not happen due to check above
        
        total_score = cost_to_c + min_future
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(c)
            
    return best_node
