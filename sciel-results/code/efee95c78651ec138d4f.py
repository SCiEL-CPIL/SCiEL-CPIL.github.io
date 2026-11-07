"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time


def heuristics(distance_matrix):
    """
    Compute a smooth topological heuristic using Delaunay triangulation.
    Uses a blended approach: base 1/distance for all edges, boosted for Delaunay edges.
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # Safety margin: stop if we've used more than 280s of the 330s budget
    if time.time() - start_time > 280:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Classical MDS to reconstruct 2D coordinates
    D = distance_matrix.copy()
    
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    
    D_sq = D ** 2
    B = -0.5 * J @ D_sq @ J
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 dimensions
    X = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    # Normalize to unit square
    min_coords = X.min(axis=0)
    max_coords = X.max(axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords < 1e-10] = 1.0
    X = (X - min_coords) / range_coords
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(X)
    except Exception:
        # Fallback to pure distance-based heuristic
        return 1.0 / (distance_matrix + 1e-9)
    
    # Check time budget
    if time.time() - start_time > 280:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Build a boolean mask of Delaunay edges
    del_edge = np.zeros((n, n), dtype=bool)
    for simplex in tri.simplices:
        i, j, k = simplex
        del_edge[i, j] = True
        del_edge[j, i] = True
        del_edge[j, k] = True
        del_edge[k, j] = True
        del_edge[i, k] = True
        del_edge[k, i] = True
    
    # Smooth heuristic: blend of 1/distance with a boost for Delaunay edges
    # base = 1/(d + eps) for all edges
    # For Delaunay edges, multiply by a boost factor
    # This creates a smooth hierarchy rather than binary in/out
    
    eps = 1e-9
    base = 1.0 / (distance_matrix + eps)
    
    # Boost factor for Delaunay edges: make them ~10x more attractive
    # This is smooth and allows non-Delaunay edges to still be used
    boost = 10.0
    heuristic = np.where(del_edge, boost * base, base)
    
    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    heuristic = np.maximum(heuristic, eps)
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    
    return heuristic
# EVOLVE-BLOCK-END
