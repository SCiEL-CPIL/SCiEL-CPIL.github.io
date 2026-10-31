"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node for a TSP tour construction.
    
    Uses a hybrid heuristic that combines:
    1. Nearest neighbor distance
    2. Distance to destination (to avoid stranding)
    3. A penalty for nodes that are far from other unvisited nodes (to keep options open)
    
    This is a constructive heuristic that aims to balance local optimality with global structure.
    """
    if len(unvisited_nodes) == 0:
        return -1
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited nodes
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Compute a penalty based on how isolated each candidate is
    # For each candidate, compute the minimum distance to other unvisited nodes
    # This penalizes picking a node that is far from the rest of the unvisited set
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited > 1:
        # For each candidate node, find the minimum distance to other unvisited nodes
        # This is a proxy for "how connected" this node is to the remaining tour
        min_dist_to_others = np.full(n_unvisited, np.inf)
        for i in range(n_unvisited):
            # Distance from candidate i to all other unvisited nodes
            dists_to_others = distance_matrix[unvisited_nodes[i], unvisited_nodes]
            # Exclude self
            dists_to_others[i] = np.inf
            min_dist_to_others[i] = np.min(dists_to_others)
    else:
        min_dist_to_others = np.array([0.0])
    
    # Normalize the components
    # Scale each component to [0, 1] range for fair combination
    def normalize(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    norm_dist_from_current = normalize(dist_from_current)
    norm_dist_to_dest = normalize(dist_to_dest)
    norm_min_dist_to_others = normalize(min_dist_to_others)
    
    # Combine with weights
    # We want to prioritize:
    # 1. Close to current (local optimality)
    # 2. Not too far from destination (avoid stranding)
    # 3. Connected to other unvisited nodes (keep options open)
    w1 = 0.5  # weight for distance from current
    w2 = 0.3  # weight for distance to destination
    w3 = 0.2  # weight for connectivity to others
    
    score = w1 * norm_dist_from_current + w2 * norm_dist_to_dest + w3 * norm_min_dist_to_others
    
    # Pick the node with the lowest score
    best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
