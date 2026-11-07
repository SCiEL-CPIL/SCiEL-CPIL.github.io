# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix based on Delaunay triangulation.
    Edges in the Delaunay triangulation get high attractiveness; others get low.
    """
    n = distance_matrix.shape[0]
    
    # Reconstruct points from the distance matrix using classical MDS
    # The distance matrix is symmetric with zero diagonal
    # Classical MDS: embed points in 2D using the distance matrix
    
    # Step 1: Double centering
    D = distance_matrix
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    
    # Step 2: Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 eigenvalues (for 2D embedding)
    # Ensure non-negative
    eigenvalues[0:2] = np.maximum(eigenvalues[0:2], 0)
    
    # Compute 2D coordinates
    points = eigenvectors[:, 0:2] * np.sqrt(eigenvalues[0:2])
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(points)
        simplices = tri.simplices
    except Exception:
        # Fallback to k-nearest neighbors if Delaunay fails
        k = 8
        heuristic = np.full((n, n), 1e-9)
        np.fill_diagonal(heuristic, 0.0)
        for i in range(n):
            dists = distance_matrix[i].copy()
            dists[i] = np.inf
            k_nearest = np.argpartition(dists, k)[:k]
            for j in k_nearest:
                d = distance_matrix[i, j]
                attractiveness = 1.0 / (d + 1e-9)
                heuristic[i, j] = max(heuristic[i, j], attractiveness)
                heuristic[j, i] = max(heuristic[j, i], attractiveness)
        heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        return heuristic
    
    # Initialize heuristic with a small base value
    heuristic = np.full((n, n), 1e-9)
    np.fill_diagonal(heuristic, 0.0)
    
    # For each edge in the Delaunay triangulation, assign high attractiveness
    # Delaunay triangulation contains triangles (simplices of size 3)
    for simplex in simplices:
        i, j, k = simplex
        # Add all three edges of the triangle
        for a, b in [(i, j), (j, k), (i, k)]:
            d = distance_matrix[a, b]
            # Higher attractiveness for closer nodes
            attractiveness = 1.0 / (d + 1e-9)
            heuristic[a, b] = max(heuristic[a, b], attractiveness)
            heuristic[b, a] = max(heuristic[b, a], attractiveness)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
