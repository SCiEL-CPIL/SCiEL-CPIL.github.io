"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP using a geometric
    approach based on classical MDS embedding, Delaunay triangulation,
    and convex hull structure.
    
    Key ideas:
    1. Reconstruct 2D coordinates from the distance matrix via classical MDS.
    2. Compute Delaunay triangulation of the embedded points.
    3. Compute the convex hull.
    4. Build a heuristic that rewards edges that are:
       - In the Delaunay triangulation (provably contains optimal TSP tour)
       - On the convex hull (hull edges are in the optimal tour)
       - Geometrically consistent (non-crossing, aligned with local structure)
    5. Within these preferred edges, rank by inverse distance squared.
    """
    n = distance_matrix.shape[0]
    D = distance_matrix.astype(np.float64)
    
    # Step 1: Classical MDS to get 2D embedding
    # D is the distance matrix; we need squared distances for MDS
    D_sq = D ** 2
    np.fill_diagonal(D_sq, 0.0)
    
    # Double centering: B = -1/2 * J * D_sq * J where J = I - (1/n) * ones
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 dimensions
    # Scale by sqrt of eigenvalues
    coords_2d = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    # Step 2: Compute Delaunay triangulation
    try:
        tri = Delaunay(coords_2d)
        delaunay_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    delaunay_edges.add((min(a, b), max(a, b)))
    except Exception:
        delaunay_edges = set()
    
    # Step 3: Compute convex hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(coords_2d)
        hull_edges = set()
        hull_vertices = hull.vertices
        for i in range(len(hull_vertices)):
            a = hull_vertices[i]
            b = hull_vertices[(i + 1) % len(hull_vertices)]
            hull_edges.add((min(a, b), max(a, b)))
    except Exception:
        hull_edges = set()
    
    # Step 4: Build base heuristic from inverse distance squared
    inv_dist_sq = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                inv_dist_sq[i, j] = 1.0 / (D[i, j] ** 2 + 1e-12)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Step 5: Build the final heuristic
    # Start with a low base value for all edges
    heuristic = np.full((n, n), 1e-6, dtype=np.float64)
    np.fill_diagonal(heuristic, 0.0)
    
    # Add Delaunay edges with a boost
    for (a, b) in delaunay_edges:
        heuristic[a, b] = max(heuristic[a, b], inv_dist_sq[a, b] * 5.0)
        heuristic[b, a] = max(heuristic[b, a], inv_dist_sq[b, a] * 5.0)
    
    # Add convex hull edges with a stronger boost
    for (a, b) in hull_edges:
        heuristic[a, b] = max(heuristic[a, b], inv_dist_sq[a, b] * 15.0)
        heuristic[b, a] = max(heuristic[b, a], inv_dist_sq[b, a] * 15.0)
    
    # Also add k-nearest neighbors (k=5) as a safety net for edges
    # that might be missed by Delaunay due to numerical issues
    try:
        tree = cKDTree(coords_2d)
        for i in range(n):
            _, nn_indices = tree.query(coords_2d[i], k=6)  # k=6 includes self
            for j in nn_indices[1:]:  # skip self
                if j < n:
                    heuristic[i, j] = max(heuristic[i, j], inv_dist_sq[i, j] * 3.0)
                    heuristic[j, i] = max(heuristic[j, i], inv_dist_sq[j, i] * 3.0)
    except Exception:
        pass
    
    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Clamp to positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
