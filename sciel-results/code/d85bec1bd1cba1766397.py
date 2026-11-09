"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Choose the next city for a TSP constructive heuristic.
    
    Uses a lookahead strategy: for each candidate, estimate the cost of going
    to that candidate and then to the nearest unvisited node (or destination).
    This gives a 2-step lookahead that balances immediate distance with future
    connectivity.
    
    Parameters:
    - current_node: int, the city we are currently at
    - destination_node: int, the final destination (city 0 in typical TSP)
    - unvisited_nodes: np.ndarray, array of unvisited city indices
    - distance_matrix: np.ndarray, n x n distance matrix
    
    Returns:
    - int: index of the next city to visit
    """
    # Handle edge cases
    if len(unvisited_nodes) == 0:
        # No unvisited nodes; this shouldn't happen normally
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Deadline guard: if we've been running too long, fall back to nearest neighbor
    # But since this is called per-step, we need to track global time
    # For now, we'll just do a fast computation
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, compute a 2-step lookahead score:
    # score[i] = dist(current, candidate_i) + dist(candidate_i, best_next_after_i)
    # where best_next_after_i is the nearest unvisited node (excluding candidate_i) 
    # or the destination if this is the last unvisited node
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited <= 1:
        return int(unvisited_nodes[np.argmin(dist_from_current)])
    
    # For each candidate i, find the minimum distance from candidate_i to any other unvisited node
    # or to the destination (if it would be the last step)
    # We can vectorize this: for each candidate, compute min distance to all other unvisited + destination
    
    # Build a matrix of distances from each unvisited node to all unvisited nodes
    # dist_unvisited[i, j] = distance from unvisited_nodes[i] to unvisited_nodes[j]
    dist_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each candidate i, the "next cost" is the min over j != i of dist_unvisited[i, j]
    # But if i is the last node, the next cost should be distance to destination
    # Actually, since we're building a tour that ends at destination_node,
    # the last unvisited node will go to destination_node.
    # So for candidate i, if it's the last one (n_unvisited == 1 after picking i),
    # then next_cost = distance from i to destination_node
    # Otherwise, next_cost = min distance from i to other unvisited nodes
    
    # Let's compute for each candidate i:
    # lookahead_score[i] = dist_from_current[i] + next_cost[i]
    
    # next_cost[i]:
    # If n_unvisited == 2, then after picking i, only one node remains, so that node goes to destination
    # Actually, the tour structure is: current -> ... -> last_unvisited -> destination
    # So when we pick a node, the "future cost" depends on how many nodes remain.
    
    # Simpler approach: use a weighted combination of:
    # 1. Distance from current to candidate
    # 2. Distance from candidate to its nearest neighbor among unvisited (excluding itself)
    # 3. Distance from candidate to destination (weighted by whether it's likely to be last)
    
    # Compute nearest neighbor distance for each candidate (excluding itself)
    # Set diagonal to inf before taking min
    dist_unvisited_diag_inf = dist_unvisited.copy()
    np.fill_diagonal(dist_unvisited_diag_inf, np.inf)
    nearest_neighbor_dist = np.min(dist_unvisited_diag_inf, axis=1)
    
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Combine: use a weighted sum
    # Weight the destination distance more heavily for candidates that are far from others
    # or use a simple average
    
    # Let's try: score = dist_from_current + 0.5 * nearest_neighbor_dist + 0.5 * dist_to_dest
    # This balances immediate cost, local connectivity, and return to start
    
    scores = dist_from_current + 0.5 * nearest_neighbor_dist + 0.5 * dist_to_dest
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
