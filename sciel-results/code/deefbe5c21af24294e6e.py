"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute a heuristic (edge-attractiveness) matrix for ACO TSP.
    
    Strategy:
    1. Reconstruct 2D coordinates via classical MDS.
    2. Build a Delaunay triangulation to identify topologically plausible edges.
    3. For Delaunay edges, use a strong inverse-distance^2 heuristic (steeper gradient).
    4. For non-Delaunay edges, use a very small heuristic to discourage long/crossing edges.
    5. Add a small global inverse-distance^2 component to ensure all edges have some baseline.
    
    Returns:
        numpy array of same shape as distance_matrix, with positive finite values.
    """
    start_time = time.time()
    deadline = 280.0  # seconds, safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    if n < 3:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Check time budget
    if time.time() - start_time > deadline:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Reconstruct 2D coordinates via classical MDS
    try:
        D2 = distance_matrix ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Take top 2 eigenvalues (largest first)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Coordinates in 2D
        coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    except Exception:
        # Fallback: pure inverse-distance heuristic
        heur = 1.0 / (distance_matrix ** 2 + 1e-9)
        np.fill_diagonal(heur, 0.0)
        heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
        return heur
    
    # Check time budget
    if time.time() - start_time > deadline:
        heur = 1.0 / (distance_matrix ** 2 + 1e-9)
        np.fill_diagonal(heur, 0.0)
        heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
        return heur
    
    # Build Delaunay triangulation
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(coords)
    except Exception:
        # Fallback: pure inverse-distance heuristic
        heur = 1.0 / (distance_matrix ** 2 + 1e-9)
        np.fill_diagonal(heur, 0.0)
        heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
        return heur
    
    # Build adjacency from Delaunay triangulation
    delaunay_adj = np.zeros((n, n), dtype=bool)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                delaunay_adj[u, v] = True
                delaunay_adj[v, u] = True
    
    # Check time budget
    if time.time() - start_time > deadline:
        heur = 1.0 / (distance_matrix ** 2 + 1e-9)
        np.fill_diagonal(heur, 0.0)
        heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
        return heur
    
    # Compute base heuristic: inverse distance squared
    base_heur = 1.0 / (distance_matrix ** 2 + 1e-9)
    
    # For Delaunay edges: strong boost with inverse distance squared
    # For non-Delaunay edges: very low value to discourage
    boost = 50.0
    damp = 1e-8
    
    heur = np.where(delaunay_adj, base_heur * boost, base_heur * damp)
    
    # Add a small global component to ensure all edges have some baseline
    # This helps with exploration while still favoring Delaunay edges
    heur += base_heur * 0.01
    
    # Ensure diagonal is 0 (no self-loops)
    np.fill_diagonal(heur, 0.0)
    
    # Ensure all values are positive and finite
    heur = np.where(np.isfinite(heur) & (heur > 0), heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
