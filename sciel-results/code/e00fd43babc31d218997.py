"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # Fallback heuristic: standard 1/distance
    fallback = 1.0 / (distance_matrix + 1e-9)
    
    try:
        # 1. Reconstruct 2D points from distance matrix using Classical MDS
        # Center the squared distance matrix
        D2 = distance_matrix ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * (J @ D2 @ J)
        
        # Eigen decomposition
        # eigh returns eigenvalues in ascending order
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # We want the top 2 eigenvalues (largest)
        # Since eigh is ascending, we take the last 2
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top 2
        top_eigvals = eigvals[:2]
        top_eigvecs = eigvecs[:, :2]
        
        # Ensure eigenvalues are non-negative (handle numerical noise)
        top_eigvals = np.maximum(top_eigvals, 0)
        
        # Reconstruct points
        points = top_eigvecs * np.sqrt(top_eigvals)
        
        # 2. Compute Delaunay Triangulation
        # Delaunay requires points to be in general position, but scipy handles some degeneracies
        tri = Delaunay(points)
        
        # 3. Build the heuristic matrix
        # Initialize with 0 (which will be converted to 1e-9 by the evaluator if non-positive)
        H = np.zeros((n, n))
        
        # Extract unique edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            edges.add((simplex[0], simplex[1]))
            edges.add((simplex[1], simplex[2]))
            edges.add((simplex[0], simplex[2]))
            
        # Assign high attractiveness to Delaunay edges
        for i, j in edges:
            # Use 1/distance for the attractiveness value
            # This preserves the local geometric preference among valid edges
            val = 1.0 / (distance_matrix[i, j] + 1e-9)
            H[i, j] = val
            H[j, i] = val
            
        # Ensure diagonal is 0 (it will be ignored by ACO as nodes are unvisited, 
        # but good practice)
        np.fill_diagonal(H, 0)
        
        return H
        
    except Exception:
        # If MDS or Delaunay fails, fall back to standard heuristic
        return fallback
# EVOLVE-BLOCK-END
