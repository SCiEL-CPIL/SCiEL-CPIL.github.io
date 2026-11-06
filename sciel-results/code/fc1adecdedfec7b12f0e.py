"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _mst_weight(nodes, distance_matrix):
    """
    Compute MST weight for a set of nodes using Prim's algorithm.
    Returns 0 if fewer than 2 nodes.
    """
    n = len(nodes)
    if n < 2:
        return 0.0
    
    # Extract submatrix
    sub = distance_matrix[np.ix_(nodes, nodes)]
    
    # Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_weight = 0.0
    
    for _ in range(n):
        # Find node with minimum distance not in MST
        in_mst_mask = ~in_mst
        if not np.any(in_mst_mask):
            break
        candidates = np.where(in_mst_mask)[0]
        min_idx = candidates[np.argmin(min_dist[candidates])]
        
        in_mst[min_idx] = True
        total_weight += min_dist[min_idx]
        
        # Update distances
        if np.any(in_mst_mask):
            min_dist[in_mst_mask] = np.minimum(min_dist[in_mst_mask], sub[min_idx, in_mst_mask])
    
    return total_weight

def _solve_mst_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    MST-based heuristic: score each candidate by the MST weight of the remaining
    unvisited nodes after visiting it, plus connection costs.
    """
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Precompute distances
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each candidate i, compute MST weight of unvisited_nodes without i
    # plus d_curr[i] + d_dest[i]
    # To speed up, we can approximate: MST of all minus the node with smallest
    # marginal contribution, but for accuracy let's compute properly for small n
    
    if n <= 25:
        # Exact MST computation for each candidate
        scores = np.zeros(n)
        for i in range(n):
            remaining = np.delete(unvisited_nodes, i)
            if len(remaining) >= 2:
                mst_w = _mst_weight(remaining, distance_matrix)
            else:
                mst_w = 0.0
            scores[i] = d_curr[i] + d_dest[i] + mst_w
        
        return int(unvisited_nodes[np.argmin(scores)])
    else:
        # Approximation for larger sets: use MST of all unvisited, 
        # and estimate marginal cost of removing each node
        mst_all = _mst_weight(unvisited_nodes, distance_matrix)
        
        # For each node, estimate how much MST weight decreases when removed
        # Approximation: min edge incident to node in MST
        # Compute MST edges
        n_nodes = len(unvisited_nodes)
        in_mst = np.zeros(n_nodes, dtype=bool)
        min_dist = np.full(n_nodes, np.inf)
        min_dist[0] = 0.0
        parent = np.full(n_nodes, -1, dtype=int)
        
        for _ in range(n_nodes):
            candidates = np.where(~in_mst)[0]
            if len(candidates) == 0:
                break
            min_idx = candidates[np.argmin(min_dist[candidates])]
            in_mst[min_idx] = True
            if parent[min_idx] >= 0:
                # Edge from parent[min_idx] to min_idx
                pass
        
        # Rebuild properly to get edges
        in_mst = np.zeros(n_nodes, dtype=bool)
        min_dist = np.full(n_nodes, np.inf)
        min_dist[0] = 0.0
        parent = np.full(n_nodes, -1, dtype=int)
        
        for _ in range(n_nodes):
            candidates = np.where(~in_mst)[0]
            if len(candidates) == 0:
                break
            min_idx = candidates[np.argmin(min_dist[candidates])]
            in_mst[min_idx] = True
            # Update
            for j in candidates:
                if d_sub[min_idx, j] < min_dist[j]:
                    min_dist[j] = d_sub[min_idx, j]
                    parent[j] = min_idx
        
        # For each node, find min incident edge in MST
        min_edge_in_mst = np.full(n_nodes, np.inf)
        for j in range(n_nodes):
            if parent[j] >= 0:
                edge_w = d_sub[parent[j], j]
                min_edge_in_mst[j] = min(min_edge_in_mst[j], edge_w)
                min_edge_in_mst[parent[j]] = min(min_edge_in_mst[parent[j]], edge_w)
        
        # Score: d_curr + d_dest + (mst_all - min_edge_in_mst)
        # The idea: removing node i reduces MST by approximately its min incident edge
        scores = d_curr + d_dest + (mst_all - min_edge_in_mst)
        
        return int(unvisited_nodes[np.argmin(scores)])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    start_time = time.time()
    
    # Deadline guard
    if time.time() - start_time > 110:
        d_curr = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(d_curr)])
        
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
        
    return _solve_mst_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
