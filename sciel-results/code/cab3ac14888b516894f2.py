import numpy as np
import time
from scipy.spatial import Delaunay, ConvexHull

def heuristics(distance_matrix):
    """
    Computes a heuristic matrix for ACO TSP.
    Uses inverse distance boosted by Delaunay Triangulation and Convex Hull membership.
    """
    start_time = time.time()
    D = np.asarray(distance_matrix)
    n = D.shape[0]
    
    # Base heuristic: inverse distance
    # Add small epsilon to avoid division by zero
    H_base = 1.0 / (D + 1e-9)
    
    try:
        # 1. Recover 2D coordinates via Classical MDS
        # D is the distance matrix. MDS works on squared distances.
        D2 = D**2
        # Centering matrix J
        J = np.eye(n) - np.ones((n, n)) / n
        # Double centered matrix B
        B = -0.5 * (J @ D2 @ J)
        
        # Eigen decomposition
        # B is symmetric, so eigh is appropriate
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top 2 dimensions
        # Clip negative eigenvalues (numerical noise)
        eigvals_top2 = np.maximum(eigvals[:2], 0.0)
        X = eigvecs[:, :2] * np.sqrt(eigvals_top2)
        
        # 2. Compute Delaunay Triangulation
        # Delaunay triangulation edges are a superset of the optimal TSP tour edges
        tri = Delaunay(X)
        is_delaunay = np.zeros((n, n), dtype=bool)
        for simplex in tri.simplices:
            i, j = simplex[0], simplex[1]
            is_delaunay[i, j] = True
            is_delaunay[j, i] = True
            
        # 3. Compute Convex Hull
        # Convex hull edges are mandatory for the TSP tour
        hull = ConvexHull(X)
        is_hull = np.zeros((n, n), dtype=bool)
        for simplex in hull.simplices:
            i, j = simplex[0], simplex[1]
            is_hull[i, j] = True
            is_hull[j, i] = True
            
        # 4. Construct Final Heuristic
        # Boost Delaunay edges by 10x
        # Boost Hull edges by an additional 10x (total 100x)
        # This strongly guides ants to geometrically consistent edges
        
        # Use multiplication to apply boosts
        # Ensure no zeros in H_base (already handled by epsilon)
        H = H_base.copy()
        H[is_delaunay] *= 10.0
        H[is_hull] *= 10.0 # Note: is_hull is a subset of is_delaunay usually, but not always? 
                           # Actually, hull edges are in Delaunay. So this multiplies them by 10 again.
                           # Total boost for hull: 100x.
                           # Total boost for non-hull Delaunay: 10x.
                           # Total boost for others: 1x.
        
        # Check for deadline (safety margin)
        if time.time() - start_time > 5.0:
            return H_base
            
        return H
        
    except Exception:
        # If MDS or Triangulation fails, fall back to simple inverse distance
        return H_base
