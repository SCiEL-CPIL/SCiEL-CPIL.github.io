"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute vertex scores for greedy Maximum Independent Set selection.
    
    Strategy: Use a combination of spectral centrality (eigenvector centrality via
    power iteration) and degree-based features. Vertices with lower eigenvector
    centrality are structurally peripheral and less entangled in core components,
    making them better candidates for early selection in a greedy MIS algorithm.
    
    We combine:
    - Negative eigenvector centrality (prefer peripheral nodes)
    - Negative degree (prefer low-degree nodes)
    - A small bonus for nodes whose neighbors have high degree (prefer leaves/pendants)
    
    The greedy decoder picks vertices in descending score order, skipping neighbors
    of already-picked vertices.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    # Handle trivial cases
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([0.0], dtype=np.float64)
    
    # Ensure symmetric (in case of minor asymmetry)
    A = 0.5 * (A + A.T)
    
    # Degree vector
    deg = A.sum(axis=1)
    
    # Compute eigenvector centrality via power iteration
    # v_{k+1} = A @ v_k, normalized
    # This converges to the principal eigenvector of A
    v = np.ones(n, dtype=np.float64)
    v = v / np.linalg.norm(v)
    
    max_iters = 200
    tol = 1e-10
    converged = False
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            v_new = np.ones(n, dtype=np.float64) / np.sqrt(n)
            break
        v_new = v_new / norm
        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            converged = True
            break
        v = v_new
    
    # Ensure non-negative (principal eigenvector should be, but numerical issues)
    v = np.maximum(v, 0.0)
    # Normalize for consistent scale
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-15:
        v = v / v_norm
    
    # Feature: minimum neighbor degree (for each vertex, the min degree among its neighbors)
    # This helps identify pendant/leaf nodes
    min_nbr_deg = np.full(n, np.inf)
    for i in range(n):
        if time.time() > deadline:
            break
        nbrs = np.where(A[i] > 0.5)[0]
        if len(nbrs) > 0:
            min_nbr_deg[i] = np.min(deg[nbrs])
        else:
            min_nbr_deg[i] = 0.0
    
    # Replace any remaining inf with 0
    min_nbr_deg = np.where(np.isinf(min_nbr_deg), 0.0, min_nbr_deg)
    
    # Outward expansiveness: for each vertex, the mean degree of its neighbors
    # High mean neighbor degree means the vertex connects to a dense core
    mean_nbr_deg = np.zeros(n)
    for i in range(n):
        if time.time() > deadline:
            break
        nbrs = np.where(A[i] > 0.5)[0]
        if len(nbrs) > 0:
            mean_nbr_deg[i] = np.mean(deg[nbrs])
    
    # Normalize features to [0, 1] range for combining
    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-15:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    # Combine features:
    # We want HIGH score for vertices we pick FIRST.
    # For greedy MIS, we want to pick vertices that:
    #   - Have low degree (less conflict)
    #   - Have low eigenvector centrality (peripheral)
    #   - Have neighbors with high degree (i.e., the vertex is a "leaf" attached to dense part)
    #     This is captured by: high min_nbr_deg means its neighbors are in the core,
    #     so removing this leaf doesn't hurt the core structure.
    
    deg_norm = normalize(deg)
    v_norm_feat = normalize(v)
    min_nbr_norm = normalize(min_nbr_deg)
    mean_nbr_norm = normalize(mean_nbr_deg)
    
    # Weighted combination. Higher score = pick earlier.
    # -deg: prefer low degree
    # -v: prefer low eigenvector centrality (peripheral)
    # +min_nbr_deg: prefer nodes whose neighbors are high-degree (pendant nodes)
    # -mean_nbr_deg: prefer nodes not connected to very dense neighborhoods
    
    w1 = 1.0   # weight for -degree
    w2 = 1.5   # weight for -eigenvector centrality
    w3 = 0.8   # weight for +min neighbor degree
    w4 = 0.3   # weight for -mean neighbor degree
    
    scores = (-w1 * deg_norm) + (-w2 * v_norm_feat) + (w3 * min_nbr_norm) + (-w4 * mean_nbr_norm)
    
    # Add tiny random tie-breaking to ensure deterministic but varied ordering
    # Actually, let's keep it deterministic for reproducibility
    # But add a very small secondary criterion: original index to break ties
    # This is already handled by stable sort in the decoder presumably
    
    return scores.astype(np.float64)
# EVOLVE-BLOCK-END
