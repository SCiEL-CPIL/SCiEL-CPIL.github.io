"""Rewritten program using principal-eigenvector 2-coloring heuristic for MIS."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 10.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def _principal_eigenvector(adjacency, n):
    """
    Compute the principal eigenvector (largest eigenvalue) of the adjacency matrix.
    Returns an array of length n, or None on failure.
    """
    if n < 3:
        return np.ones(n)
    
    A = csr_matrix(adjacency.astype(np.float64))
    
    try:
        # Use 'LA' for largest algebraic eigenvalue
        vals, vecs = eigsh(A, k=1, which='LA', maxiter=n*20)
        if len(vals) >= 1:
            v = vecs[:, 0]
            # Normalize to have zero mean and unit norm
            v = v - v.mean()
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                v = v / norm
            return v
    except Exception:
        pass
    
    # Fallback: try 'LM' for largest magnitude
    try:
        vals, vecs = eigsh(A, k=1, which='LM', maxiter=n*20)
        if len(vals) >= 1:
            v = vecs[:, 0]
            v = v - v.mean()
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                v = v / norm
            return v
    except Exception:
        pass
    
    return None


def priority(adjacency):
    """
    Compute priority scores using principal eigenvector 2-coloring heuristic.
    
    Strategy:
    1. Compute the principal eigenvector of the adjacency matrix.
    2. Use the sign of the eigenvector components to define two color classes.
    3. Prefer vertices in the larger color class (which approximates a large independent set).
    4. Within each color class, prefer lower-degree vertices and vertices connected to
       high-degree neighbors (hubs), as pruning hubs early helps the greedy decoder.
    """
    n = adjacency.shape[0]
    adjacency = np.asarray(adjacency, dtype=bool)
    
    degree = adjacency.sum(axis=1).astype(np.float64)
    
    # Get principal eigenvector
    pe = _principal_eigenvector(adjacency, n)
    
    if pe is None:
        # Fallback: just use negative degree
        return -degree
    
    if _is_deadline_exceeded():
        return -degree
    
    # Determine color classes based on sign of eigenvector
    # Positive component -> color 0, negative -> color 1
    color = (pe > 0).astype(np.int32)
    color0_count = np.sum(color == 0)
    color1_count = np.sum(color == 1)
    
    # The larger color class is preferred (approximates larger independent set)
    # Assign a bonus to vertices in the larger class
    if color0_count >= color1_count:
        color_bonus = np.where(color == 0, 1.0, 0.0)
    else:
        color_bonus = np.where(color == 1, 1.0, 0.0)
    
    # Normalize degree to [0, 1]
    d_min, d_max = degree.min(), degree.max()
    if d_max - d_min > 1e-12:
        degree_norm = (degree - d_min) / (d_max - d_min)
    else:
        degree_norm = np.zeros(n)
    
    # Compute neighbor-degree-sum: sum of degrees of neighbors
    # This rewards vertices connected to high-degree nodes (hubs)
    neighbor_degree_sum = adjacency.astype(np.float64) @ degree
    
    # Normalize neighbor-degree-sum to [0, 1]
    nds_min, nds_max = neighbor_degree_sum.min(), neighbor_degree_sum.max()
    if nds_max - nds_min > 1e-12:
        nds_norm = (neighbor_degree_sum - nds_min) / (nds_max - nds_min)
    else:
        nds_norm = np.zeros(n)
    
    # Use the magnitude of the eigenvector component as a confidence signal
    # Vertices with larger |pe[i]| are more "committed" to their color class
    pe_mag = np.abs(pe)
    pe_mag_min, pe_mag_max = pe_mag.min(), pe_mag.max()
    if pe_mag_max - pe_mag_min > 1e-12:
        pe_mag_norm = (pe_mag - pe_mag_min) / (pe_mag_max - pe_mag_min)
    else:
        pe_mag_norm = np.zeros(n)
    
    # Combined score:
    # - Strong bonus for being in the larger color class
    # - Reward for high confidence (large |pe|)
    # - Reward for connecting to hubs (high neighbor-degree-sum)
    # - Penalize high degree
    # 
    # The key insight: preferentially selecting vertices from the dominant color class
    # approximates a large independent set in sparse graphs.
    
    w_color = 3.0      # Strong weight on color class membership
    w_confidence = 1.0  # Weight on eigenvector magnitude
    w_hub = 1.5         # Weight on neighbor-degree-sum (hub connection)
    w_degree = 0.5      # Weight on degree penalty
    
    score = (w_color * color_bonus + 
             w_confidence * pe_mag_norm + 
             w_hub * nds_norm - 
             w_degree * degree_norm)
    
    return score
# EVOLVE-BLOCK-END
