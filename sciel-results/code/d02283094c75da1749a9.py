import numpy as np
import time
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    
    Args:
        distance_matrix: (n, n) matrix of Euclidean distances
        coordinates: (n, 2) array of coordinates in unit square
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity (50)
    
    Returns:
        (n, n) edge-attractiveness matrix where larger values are more attractive.
        Non-positive entries become 1e-9.
    """
    d = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    n = d.shape[0]
    
    # Ensure shape
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    # Start with base inverse distance heuristic
    heuristic = np.zeros_like(d)
    mask = d > 1e-12
    heuristic[mask] = 1.0 / (d[mask] ** 2.5)
    
    # Demand factor: favor edges to high-demand nodes
    demand_factor = 1.0 + demands / capacity
    heuristic = heuristic * demand_factor[np.newaxis, :]
    
    # Handle non-positive or non-finite values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Graph Laplacian Spectral Clustering heuristic
    deadline = time.time() + 280  # Safety margin before 330s limit
    
    # Build a distance-weighted graph where edge weights are inverse distances
    # (higher weight = closer = more likely to be in same cluster)
    # Use the customer subgraph (nodes 1 to n-1)
    customer_indices = np.arange(1, n)
    num_customers = n - 1
    
    # Create weighted adjacency matrix for customers
    # Weight = 1/distance (higher for closer nodes)
    cust_dist = d[1:, 1:]  # (num_customers, num_customers)
    
    # Avoid division by zero
    safe_dist = np.where(cust_dist > 1e-12, cust_dist, 1e12)
    weights = 1.0 / safe_dist
    np.fill_diagonal(weights, 0)
    
    # Build sparse Laplacian
    L = sparse_laplacian(csr_matrix(weights), normed=False)
    
    # Compute Fiedler vector (2nd smallest eigenvector)
    try:
        # Use k=2 to get the two smallest eigenvalues/eigenvectors
        # eigsh returns eigenvalues in ascending order
        eigvals, eigvecs = eigsh(L, k=2, which='SM')
        fiedler_vector = eigvecs[:, 1]  # 2nd eigenvector
    except Exception:
        # Fallback: use a simpler clustering based on coordinates
        fiedler_vector = np.zeros(num_customers)
        if num_customers > 0:
            # Use x-coordinate as a proxy for clustering
            fiedler_vector = coords[1:, 0] - coords[1:, 0].mean()
    
    # Normalize Fiedler vector
    fv_norm = np.linalg.norm(fiedler_vector)
    if fv_norm > 1e-12:
        fiedler_vector = fiedler_vector / fv_norm
    
    # Compute pairwise "cluster affinity" based on Fiedler vector
    # Nodes with similar Fiedler values are likely in the same cluster
    # Use Gaussian kernel: affinity = exp(-gamma * (fv_i - fv_j)^2)
    fv_diff = np.abs(fiedler_vector[:, np.newaxis] - fiedler_vector[np.newaxis, :])
    # Choose gamma based on median difference
    nonzero_diffs = fv_diff[fv_diff > 1e-12]
    if len(nonzero_diffs) > 0:
        gamma = 1.0 / (np.median(nonzero_diffs) ** 2 + 1e-12)
    else:
        gamma = 1.0
    
    affinity = np.exp(-gamma * fv_diff ** 2)
    np.fill_diagonal(affinity, 0)
    
    # Boost heuristic for edges within clusters (high affinity)
    # and suppress edges between clusters (low affinity)
    # Map customer indices back to full matrix indices
    # customer i in affinity corresponds to node i+1 in full matrix
    for i in range(num_customers):
        for j in range(i + 1, num_customers):
            aff = affinity[i, j]
            # Boost factor: higher affinity -> higher boost
            # Use a multiplicative factor that ranges from ~0.5 to ~3.0
            boost = 0.5 + 2.5 * aff
            heuristic[i + 1, j + 1] *= boost
            heuristic[j + 1, i + 1] *= boost
    
    # Also boost edges from depot to customers based on their cluster
    # Customers in well-separated clusters may need separate routes
    # Boost depot edges to customers with extreme Fiedler values (cluster centers)
    fv_abs = np.abs(fiedler_vector)
    fv_max = np.max(fv_abs) if np.max(fv_abs) > 1e-12 else 1.0
    for i in range(num_customers):
        # Customers near cluster boundaries (low |fv|) are more likely to be 
        # connected to other clusters, so slightly reduce their depot edge
        # Customers at cluster extremes (high |fv|) are more likely to be 
        # route starts/ends, so boost their depot edge
        depot_boost = 1.0 + 1.5 * (fv_abs[i] / fv_max)
        heuristic[0, i + 1] *= depot_boost
        heuristic[i + 1, 0] *= depot_boost
    
    # Final cleanup
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
