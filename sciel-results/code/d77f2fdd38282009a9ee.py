# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        >= 0.5 -> route the demand counter-clockwise
        < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n_nodes = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute the set of edges used by each demand in each direction
    # Edge e joins nodes e and (e+1) mod 10
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod 10)
    # Counter-clockwise: all other edges
    
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        # Clockwise edges
        if s <= t:
            edges = list(range(s, t))
        else:
            edges = list(range(s, n_nodes)) + list(range(0, t))
        cw_edges.append(edges)
        
        # Counter-clockwise edges (complement)
        all_edges = set(range(n_nodes))
        cw_set = set(edges)
        ccw_list = [e for e in range(n_nodes) if e not in cw_set]
        ccw_edges.append(ccw_list)
    
    # Precompute for each demand and direction, the edge indices
    # We'll represent a routing as a binary vector: 0 = CW, 1 = CCW
    # For each combination, compute edge loads
    
    # To speed up, we can use bitmasks or vectorized numpy
    # 2^15 = 32768 combinations, which is small enough for brute force
    
    best_load = float('inf')
    best_routing = np.zeros(D)
    
    # We can use a smarter approach: iterate over all 2^15 combinations
    # For each, compute max edge load
    
    # Precompute edge contributions for each demand in each direction
    # edge_load_matrix[d][e] = weight if demand d uses edge e in that direction
    
    cw_matrix = np.zeros((D, n_nodes))
    ccw_matrix = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        for e in cw_edges[i]:
            cw_matrix[i, e] = w
        for e in ccw_edges[i]:
            ccw_matrix[i, e] = w
    
    # Now for each routing (binary vector r of length D),
    # edge loads = sum over d of (r[d] * ccw_matrix[d] + (1-r[d]) * cw_matrix[d])
    
    # Let's vectorize: for a given routing vector r (0/1),
    # loads = r @ ccw_matrix.T + (1-r) @ cw_matrix.T
    #       = r @ (ccw_matrix.T - cw_matrix.T) + cw_matrix.T @ ones
    
    # Actually: loads = cw_matrix.T @ (1-r) + ccw_matrix.T @ r
    #           = cw_matrix.T @ ones - cw_matrix.T @ r + ccw_matrix.T @ r
    #           = cw_matrix.T @ ones + r @ (ccw_matrix.T - cw_matrix.T)
    
    base_loads = cw_matrix.sum(axis=0)  # loads if all CW
    delta_matrix = ccw_matrix - cw_matrix  # change in load per edge when switching d to CCW
    
    # For routing r (binary), loads = base_loads + delta_matrix @ r
    # where delta_matrix is (n_nodes, D) and r is (D,)
    
    delta_matrix_T = delta_matrix.T  # (D, n_nodes)
    
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Brute force all 2^15 combinations
    # We can use numpy to vectorize over many routings at once
    # But 32768 is small, let's just iterate
    
    # Actually, let's use a smarter approach: since 2^15 = 32768, we can
    # iterate through all and compute loads quickly
    
    for bits in range(2**D):
        if time.time() > deadline - 5:
            break
        
        # Convert bits to binary vector
        r = np.array([(bits >> i) & 1 for i in range(D)], dtype=np.float64)
        
        # Compute loads
        loads = base_loads + delta_matrix @ r
        max_load = np.max(loads)
        
        if max_load < best_load:
            best_load = max_load
            best_routing = r.copy()
    
    return best_routing

# EVOLVE-BLOCK-END
