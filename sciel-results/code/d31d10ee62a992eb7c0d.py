import numpy as np
from scipy.spatial import Delaunay

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    if n < 3:
        return 1.0 / (distance_matrix + 1e-9)
    
    # 1. Reconstruct points using Classical MDS
    D = distance_matrix
    # Centering matrix
    J = np.eye(n) - np.ones((n, n)) / n
    # Double centering
    B = -0.5 * J @ (D**2) @ J
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2 components
    # Clip negative eigenvalues to 0
    eigvals = np.maximum(eigvals, 0)
    
    # Points in 2D
    points = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    
    # 2. Compute Delaunay Triangulation
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback if triangulation fails (e.g., degenerate points)
        return 1.0 / (D + 1e-9)
    
    # 3. Create Delaunay Edge Mask
    del_mask = np.zeros((n, n), dtype=bool)
    for simplex in tri.simplices:
        i, j, k = simplex
        del_mask[i, j] = True
        del_mask[j, i] = True
        del_mask[j, k] = True
        del_mask[k, j] = True
        del_mask[k, i] = True
        del_mask[i, k] = True
        
    # 4. Construct Heuristic
    # Base: 1/D
    base = 1.0 / (D + 1e-9)
    
    # Boost Delaunay edges
    boost_factor = 100.0
    heuristic = base * (1.0 + (boost_factor - 1.0) * del_mask)
    
    # Ensure positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
