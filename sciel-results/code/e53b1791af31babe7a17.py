"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return destination_node
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # 1-step lookahead: for each candidate, cost = dist(current, cand) + 
    #   (dist(cand, dest) if cand is last, else min dist from cand to other unvisited)
    # For efficiency, compute the min distance from each candidate to other unvisited nodes.
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For the last node, the "future cost" is dist to destination
    # For others, it's the min distance to any other unvisited node
    # This is O(n^2) but n <= 49, so it's fine
    
    # Create a mask: for each candidate i, we want min dist to unvisited[j] where j != i
    # Use a matrix of distances among unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i, set diagonal to inf, then take min along axis 1
    sub_dist_copy = sub_dist.copy()
    np.fill_diagonal(sub_dist_copy, np.inf)
    min_future_cost = np.min(sub_dist_copy, axis=1)
    
    # For the last candidate (index n_unvisited-1 in the tour being built), 
    # the future cost should be distance to destination, not to other unvisited.
    # But we don't know which will be last. A simpler approximation:
    # Use min(dist_to_other_unvisited, dist_to_destination) as a proxy for future cost.
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    future_cost = np.minimum(min_future_cost, dist_to_dest)
    
    # Total cost = dist_from_current + future_cost
    total_cost = dist_from_current + future_cost
    
    return int(unvisited_nodes[np.argmin(total_cost)])
# EVOLVE-BLOCK-END
