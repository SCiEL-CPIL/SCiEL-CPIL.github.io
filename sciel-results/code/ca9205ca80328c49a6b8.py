# EVOLVE-BLOCK-START
import numpy as np


def _get_min_mst_edge_estimate(remaining_nodes, distance_matrix):
    """
    Estimate the residual tour cost based on the minimum edge weight in the MST 
    of the remaining nodes. 
    Since the minimum edge of a complete graph is always in the MST, we find the 
    minimum distance between any pair of remaining nodes. We scale this by the 
    number of edges in the MST (k-1) to approximate the total MST weight lower bound.
    """
    k = len(remaining_nodes)
    if k <= 1:
        return 0.0
    
    min_edge = float('inf')
    # Find minimum edge in the subgraph of remaining nodes
    # O(k^2)
    for i in range(k):
        u = remaining_nodes[i]
        for j in range(i + 1, k):
            v = remaining_nodes[j]
            d = distance_matrix[u, v]
            if d < min_edge:
                min_edge = d
                
    # Approximation of MST weight: min_edge * (number of edges)
    # This is a lower bound on the MST weight.
    return min_edge * (k - 1)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that minimizes the sum of 
    the immediate distance and an MST-based residual estimate derived from the 
    minimum edge weight in the MST of the remaining unvisited nodes.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    best_score = float('inf')
    best_node = -1
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate cost: distance from current to candidate
        immediate_dist = distance_matrix[current_node, candidate]
        
        # Determine remaining unvisited nodes
        remaining_indices = [j for j in range(n_unvisited) if j != i]
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        # Residual cost estimate based on min edge in MST of remaining nodes
        residual_est = _get_min_mst_edge_estimate(remaining_nodes, distance_matrix)
        
        # Add connection costs to ensure the candidate connects well to the rest and destination
        # This prevents choosing a node that is far from the cluster even if the cluster is tight
        if len(remaining_nodes) > 0:
            min_dist_cand_rem = np.min(distance_matrix[candidate, remaining_nodes])
            min_dist_dest_rem = np.min(distance_matrix[destination_node, remaining_nodes])
            residual_est += min_dist_cand_rem + min_dist_dest_rem
        else:
            # If no remaining nodes, we just need to go to destination
            residual_est = distance_matrix[candidate, destination_node]
            
        total_score = immediate_dist + residual_est
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
