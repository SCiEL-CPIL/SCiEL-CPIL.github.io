# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 130.0  # Safety margin under 150s limit

def _is_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT_LIMIT

def _mst_weight_vectorized(nodes, distance_matrix):
    """
    Vectorized Prim's MST weight for a subset of nodes.
    O(n^2) but uses NumPy vector ops for speed.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Extract sub-matrix
    sub = distance_matrix[np.ix_(nodes, nodes)]
    
    # Prim's algorithm
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = sub[0].copy()  # min distance to MST for each node
    
    total = 0.0
    for _ in range(1, n):
        # Mask already-in-MST nodes with inf
        min_dist_masked = np.where(in_mst, np.inf, min_dist)
        u = int(np.argmin(min_dist_masked))
        total += min_dist[u]
        in_mst[u] = True
        # Update: min(current, sub[u])
        np.minimum(min_dist, sub[u], out=min_dist)
    
    return total

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a fast MST-incremental + regret heuristic.
    
    Key insight: MST(U \ {u}) = MST(U) + min_{v in U\{u}} d(u,v) when u is a leaf.
    Since we're constructing a tour by peeling nodes, this holds exactly.
    """
    if _is_timeout():
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)
    
    # Pre-compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Sub-matrix of unvisited-to-unvisited distances
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute MST weight of ALL unvisited nodes (one vectorized call)
    mst_all = _mst_weight_vectorized(unvisited_nodes, distance_matrix)
    
    # For each candidate u, the MST of remaining = mst_all + min_edge_from_u_to_remaining
    # This is exact when u is a leaf of the MST (which it is in our peeling context)
    # min_edge_from_u_to_remaining = min over v != u of d(u, v)
    # Set diagonal to inf so argmin skips self
    d_sub_no_self = d_sub.copy()
    np.fill_diagonal(d_sub_no_self, np.inf)
    min_edge_from_u = np.min(d_sub_no_self, axis=1)
    
    # MST of remaining after removing u
    mst_remaining = mst_all + min_edge_from_u
    
    # Soft-max distance from each u to farthest other (smooth surrogate)
    d_sub_softmax = d_sub.copy()
    np.fill_diagonal(d_sub_softmax, 0)
    temperature = 5.0
    d_scaled = d_sub_softmax / temperature
    d_max = np.max(d_scaled, axis=1, keepdims=True)
    d_exp = np.exp(d_scaled - d_max)
    np.fill_diagonal(d_exp, 0)
    d_sum = np.sum(d_exp, axis=1)
    soft_max_dist = temperature * (d_max.squeeze(axis=1) + np.log(d_sum + 1e-10))
    
    # Destination-aware term: cost to close the tour
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Composite score for each candidate
    # immediate + alpha * MST_remaining + beta * soft_max + gamma * dist_to_dest
    alpha = 1.0
    beta = 0.05
    gamma = 0.3
    
    scores = dist_from_current + alpha * mst_remaining + beta * soft_max_dist + gamma * dist_to_dest
    
    # Pick the best
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
