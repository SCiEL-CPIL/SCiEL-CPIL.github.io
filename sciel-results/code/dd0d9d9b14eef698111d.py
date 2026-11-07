import numpy as np
import scipy.spatial
import scipy.linalg

def heuristics(distance_matrix):
    """
    Returns a heuristic matrix based on Delaunay triangulation of the points.
    """
    n = distance_matrix.shape[0]
    
    # 1. Reconstruct 2D points from distance matrix using Classical MDS
    # D is n x n
    D2 = distance_matrix ** 2
    
    # Centering matrix J = I - (1/n) * ones
    # B = -0.5 * J * D2 * J
    # To avoid constructing J explicitly (n is small, 50, so it's fine either way)
    # J * D2 * J can be computed as D2 - row_means - col_means + grand_mean
    
    row_means = np.sum(D2, axis=1) / n
    col_means = np.sum(D2, axis=0) / n
    grand_mean = np.sum(D2) / (n * n)
    
    B = -0.5 * (D2 - row_means[:, None] - col_means[None, :] + grand_mean)
    
    # Eigen decomposition
    # We only need the top 2 eigenvalues/vectors
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2
    # If eigenvalues are negative (due to numerical noise), clip to 0
    eigvals = np.clip(eigvals, 0, None)
    
    # Coordinates = eigvecs * sqrt(eigvals)
    # We want 2D, so take first 2 columns
    if n >= 2:
        points = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    else:
        # Fallback if n is too small (not the case here, n=50)
        return 1.0 / (distance_matrix + 1e-9)
        
    # 2. Compute Delaunay Triangulation
    try:
        tri = scipy.spatial.Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    u, v = simplex[i], simplex[j]
                    edges.add((u, v))
                    edges.add((v, u))
    except Exception:
        # Fallback to standard heuristic if triangulation fails
        return 1.0 / (distance_matrix + 1e-9)

    # 3. Construct Heuristic Matrix
    heuristic = np.full((n, n), 1e-9)
    
    for i, j in edges:
        # Use inverse distance for valid edges to prefer shorter connections
        # among the topologically valid ones
        heuristic[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
        
    # Ensure diagonal is 0 or 1? The prompt says "Heuristic entries that are not positive become 1e-9".
    # Usually diagonal is ignored in selection. Let's set diagonal to 1.0 to be safe/positive,
    # or 0 if the algorithm handles it. The seed used 1/(d+eps) which is large on diagonal.
    # However, ants don't pick visited nodes. So diagonal value doesn't matter much.
    # Let's keep it consistent with "large is attractive".
    np.fill_diagonal(heuristic, 1.0)
    
    return heuristic
