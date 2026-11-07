"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _mds_2d(dist_matrix):
    """
    Recover 2D coordinates from a distance matrix using Classical MDS.
    """
    n = dist_matrix.shape[0]
    # Centering matrix
    J = np.eye(n) - np.ones((n, n)) / n
    # Double centering
    B = -0.5 * J @ (dist_matrix ** 2) @ J
    # Eigen decomposition
    # np.linalg.eigh returns eigenvalues in ascending order
    eigvals, eigvecs = np.linalg.eigh(B)
    # We want the top 2 eigenvalues (largest)
    # Reverse the order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take the top 2 eigenvectors
    # Scale by the square root of the eigenvalues
    # Clamp eigenvalues to 0 to avoid sqrt of negative numbers due to numerical errors
    vals = np.sqrt(np.maximum(eigvals[:2], 0))
    coords = eigvecs[:, :2] * vals
    return coords

def heuristics(distance_matrix):
    """
    Compute the heuristic matrix for ACO.
    Uses MDS to recover coordinates, identifies convex hull edges,
    and boosts the heuristic for these edges.
    """
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin for 300s limit
    
    # 1. Base Heuristic: Inverse Distance
    # Add small epsilon to avoid division by zero
    eta = 1.0 / (distance_matrix + 1e-9)
    
    # 2. Structural Prior: Convex Hull Boosting
    try:
        # Recover coordinates
        coords = _mds_2d(distance_matrix)
        
        # Compute Convex Hull
        # If all points are collinear or identical, ConvexHull might raise QhullError
        hull = ConvexHull(coords)
        
        # In 2D, hull.simplices contains the edges of the convex hull
        hull_edges = hull.simplices
        
        # Create a boolean mask for hull edges
        is_hull_edge = np.zeros_like(distance_matrix, dtype=bool)
        for i, j in hull_edges:
            is_hull_edge[i, j] = True
            is_hull_edge[j, i] = True
            
        # Boost the heuristic for hull edges
        # Factor of 10.0 makes them significantly more attractive
        eta[is_hull_edge] *= 10.0
        
    except Exception:
        # If MDS or Hull fails, fall back to base heuristic
        pass

    # 3. Sanitize Heuristic
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # Check time (though this operation is very fast)
    if time.time() > deadline:
         # Return best effort (which is just the base heuristic if it failed)
         pass

    return eta
# EVOLVE-BLOCK-END
