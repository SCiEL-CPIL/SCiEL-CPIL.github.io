import numpy as np
from scipy.spatial import Delaunay
import time

def heuristics(distance_matrix):
    d = distance_matrix
    n = d.shape[0]
    
    # Deadline guard: return a safe heuristic if we're running out of time
    start_time = time.time()
    DEADLINE = 280.0  # Leave 20s margin before 300s kill
    
    # Base heuristic: inverse squared distance (always valid)
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e9)  # Avoid division by zero on diagonal
    H_base = 1.0 / (d_safe ** 2)
    np.fill_diagonal(H_base, 0.0)
    
    # Try to reconstruct coordinates and use Delaunay triangulation
    try:
        if time.time() - start_time > DEADLINE:
            return H_base
        
        # Classical MDS to reconstruct 2D coordinates
        D2 = d ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * (J @ D2 @ J)
        
        # Eigen decomposition - get top 2 eigenvalues/vectors
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Check if the reconstruction is reasonable (top 2 eigenvalues should dominate)
        if eigvals[0] < 1e-10 or eigvals[1] < 1e-10:
            # Degenerate case, fall back to base heuristic
            return H_base
        
        # Reconstruct 2D coordinates
        coords = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
        
        if time.time() - start_time > DEADLINE:
            return H_base
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(coords)
        except Exception:
            # Delaunay failed, use base heuristic
            return H_base
        
        # Build adjacency from Delaunay edges
        delaunay_adj = np.zeros((n, n), dtype=bool)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    delaunay_adj[a, b] = True
                    delaunay_adj[b, a] = True
        
        # Also add a few nearest neighbors beyond Delaunay for robustness
        # Find 3 nearest neighbors for each node (excluding self)
        d_for_sort = d.copy()
        np.fill_diagonal(d_for_sort, np.inf)
        
        # Get 3 nearest neighbors
        k = 3
        indices = np.argpartition(d_for_sort, k, axis=1)[:, :k]
        
        knn_adj = np.zeros((n, n), dtype=bool)
        rows = np.arange(n)[:, np.newaxis]
        knn_adj[rows, indices] = True
        knn_adj = knn_adj | knn_adj.T
        
        # Combined adjacency: Delaunay edges get high weight, k-NN get medium
        # Non-adjacent edges get low weight (but still positive)
        combined_adj = delaunay_adj.astype(float) + 0.5 * knn_adj.astype(float)
        
        # Smooth surrogate: use log-sum-exp style weighting
        # For edges in the adjacency graph, boost the heuristic
        # Use a smooth transition based on whether edge is in graph
        
        # Scale the base heuristic and add adjacency boost
        # The boost should be significant but not overwhelming
        boost_factor = 10.0
        H = H_base * (1.0 + boost_factor * combined_adj)
        
        # Ensure all values are positive and finite
        H[~np.isfinite(H)] = 1e-9
        H[H <= 0] = 1e-9
        
        # Symmetrize to be safe
        H = (H + H.T) / 2.0
        
        if time.time() - start_time > DEADLINE:
            return H_base
        
        return H
        
    except Exception:
        # Any failure, return the safe base heuristic
        H_base[~np.isfinite(H_base)] = 1e-9
        H_base[H_base <= 0] = 1e-9
        return H_base
