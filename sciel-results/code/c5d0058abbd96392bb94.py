import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to nearest neighbor to ensure completion
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Get distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # We want to minimize: dist(current, c) + min_{n in unvisited, n != c} dist(c, n)
    # If this is the second to last node, we should also consider the return to destination?
    # Let's stick to the 2-step lookahead for consistency, but for the last two, 
    # the "next" node is the last one, so we add dist(c, last_node).
    
    # To do this efficiently:
    # For each candidate c_i, we need min(dist(c_i, c_j)) for j != i.
    
    # Create a submatrix of distances among unvisited nodes
    # unvisited_nodes is an array of indices
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i (candidate c_i), we want the minimum value in that row, 
    # but we must exclude the diagonal element (distance to self, which is 0).
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(sub_matrix, np.inf)
    
    # min_dist_to_others[i] is the min distance from c_i to any other unvisited node
    min_dist_to_others = np.min(sub_matrix, axis=1)
    
    # Total estimated cost for each candidate
    # cost_i = dists_from_current[i] + min_dist_to_others[i]
    
    # However, if n_unvisited == 2, min_dist_to_others[i] is just the distance to the other node.
    # This effectively minimizes d(u, c1) + d(c1, c2) vs d(u, c2) + d(c2, c1).
    # Since d(c1, c2) == d(c2, c1), it picks the one closer to u.
    # This is correct for the penultimate step.
    
    # What about the final step? 
    # When n_unvisited == 1, we return the only node.
    # The cost of that final step d(last, dest) is not considered in the selection of the penultimate node 
    # unless we incorporate it.
    
    # Let's refine: 
    # If n_unvisited > 2: cost = d(u, c) + min_dist_to_others[c]
    # If n_unvisited == 2: cost = d(u, c) + d(c, other_node)
    #   Note: min_dist_to_others handles this correctly because the only other node is the one left.
    #   So the formula holds.
    
    # Is it better to add a term for the destination?
    # A common heuristic is d(u, c) + d(c, dest).
    # Let's try combining: d(u, c) + 0.5 * min_dist_to_others[c] + 0.5 * d(c, dest)?
    # Or just d(u, c) + min_dist_to_others[c] is a standard "lookahead".
    
    # Let's test the simple 2-step lookahead first.
    total_costs = dists_from_current + min_dist_to_others
    
    best_idx = np.argmin(total_costs)
    
    return int(unvisited_nodes[best_idx])
