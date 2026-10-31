import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0

def _potential_field_score(candidates, current_node, destination_node, distance_matrix):
    """
    Score candidates using a potential field model:
    - Attractive: distance from current node
    - Repulsive: inverse distance sum to other unvisited nodes (encourages spatial spread)
    - Terminal: distance to destination
    
    The repulsive term prevents clustering and ensures good spatial coverage.
    """
    n = len(candidates)
    if n == 0:
        return np.array([])
    
    # Distance from current to each candidate
    d_cur = distance_matrix[current_node, candidates]
    
    # Distance from each candidate to destination
    d_dest = distance_matrix[candidates, destination_node]
    
    # Compute repulsive potential for each candidate
    # For candidate i, repulsive = sum over j != i of 1/(d(c_i, c_j) + epsilon)
    # This is high when candidate is close to many others (crowded), low when isolated
    
    scores = np.zeros(n)
    
    if n > 1:
        # Compute pairwise distances among candidates
        # dist_matrix[i,j] = distance between candidate i and candidate j
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = distance_matrix[candidates[i], candidates[j]]
                dist_mat[i, j] = d
                dist_mat[j, i] = d
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        inv_dist = 1.0 / (dist_mat + epsilon)
        np.fill_diagonal(inv_dist, 0.0)
        
        # Repulsive potential for each candidate = sum of inverse distances to others
        repulsive = inv_dist.sum(axis=1)
        
        # Normalize repulsive term to be comparable to distance terms
        # Scale by a factor that makes it meaningful relative to tour length
        # For 50 cities in unit square, typical distance ~0.1, tour length ~15-20
        # Repulsive sum for n candidates ~ n * 1/0.1 = 10n, so scale down
        scale = 0.1 / max(1, n)
        repulsive_scaled = repulsive * scale
        
        # Terminal cost: how expensive is it to go from candidate to destination?
        # We want to avoid candidates that are far from destination if we're late in the tour
        # But we don't know how late we are, so use a soft penalty
        terminal_penalty = d_dest * 0.1
    else:
        repulsive_scaled = np.zeros(n)
        terminal_penalty = d_dest * 0.1
    
    # Combined score: minimize this
    # Primary: distance from current
    # Secondary: repulsive potential (lower is better = more isolated)
    # Tertiary: terminal distance (light weight)
    scores = d_cur - repulsive_scaled + terminal_penalty
    
    return scores

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time, _LIMIT
    
    if (time.time() - _start_time) > _LIMIT:
        # Fallback: Nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        # Exact for 2 nodes
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        
        if cost1 <= cost2:
            return c1
        else:
            return c2

    # General case: Potential Field Heuristic
    unvisited_arr = np.asarray(unvisited_nodes)
    
    # Compute scores for all candidates
    scores = _potential_field_score(unvisited_arr, current_node, destination_node, distance_matrix)
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_arr[best_idx])
