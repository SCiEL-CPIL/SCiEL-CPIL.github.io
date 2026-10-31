import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill

def _mst_weight(nodes, dist_matrix):
    """
    Calculate MST weight of the subgraph induced by 'nodes' using Prim's algorithm.
    nodes: array of node indices
    dist_matrix: full distance matrix
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Map node indices to 0..n-1 for local indexing
    # We will use a visited array and a min_dist array
    # Start from nodes[0]
    
    # To avoid O(N^2) mapping overhead, we can just use the global indices
    # but we need to ensure we only pick from 'nodes'.
    
    visited = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    
    total_weight = 0.0
    
    for _ in range(n):
        # Find unvisited node with min_dist
        # Mask visited
        available = np.where(~visited, min_dist, np.inf)
        u_idx = np.argmin(available)
        u_node = nodes[u_idx]
        
        total_weight += min_dist[u_idx]
        visited[u_idx] = True
        
        # Update min_dist for neighbors
        # Get distances from u_node to all nodes in 'nodes'
        # dist_matrix[u_node, nodes] gives an array of length n
        dists = dist_matrix[u_node, nodes]
        
        # Update min_dist where not visited
        # We only care about unvisited nodes
        # min_dist = np.minimum(min_dist, dists) is too broad if we don't mask visited?
        # Actually, we can just update and let the argmin handle it, or mask.
        # Masking is safer.
        update_mask = ~visited
        # But we just set visited[u_idx] = True, so u_idx is now True.
        # We want to update others.
        
        # Efficient update:
        # min_dist[update_mask] = np.minimum(min_dist[update_mask], dists[update_mask])
        
        # However, creating the mask every time is O(N).
        # Since N=50, it's fine.
        
        min_dist[~visited] = np.minimum(min_dist[~visited], dists[~visited])
        
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        c1 = unvisited_nodes[0]
        c2 = unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1) if cost1 <= cost2 else int(c2)

    # Calculate scores for each candidate
    # Score(u) = dist(current, u) + dist(u, destination) + MST(unvisited \ {u})
    
    best_score = np.inf
    best_node = unvisited_nodes[0]
    
    # Pre-extract distances for speed
    dist_current = distance_matrix[current_node, unvisited_nodes]
    dist_dest = distance_matrix[destination_node, unvisited_nodes]
    
    for i in range(n_unvisited):
        u = unvisited_nodes[i]
        
        # Base costs
        cost = dist_current[i] + dist_dest[i]
        
        # Calculate MST of remaining nodes
        # Create a copy of unvisited_nodes excluding u
        # Using list slicing or boolean mask
        # Boolean mask is O(N)
        mask = np.ones(n_unvisited, dtype=bool)
        mask[i] = False
        remaining = unvisited_nodes[mask]
        
        mst_w = _mst_weight(remaining, distance_matrix)
        
        total_score = cost + mst_w
        
        if total_score < best_score:
            best_score = total_score
            best_node = u
            
    return int(best_node)
