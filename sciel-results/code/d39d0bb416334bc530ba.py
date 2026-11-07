"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    if n <= 1:
        return np.ones((n, n))
    
    # Extract customer data (exclude depot at index 0)
    cust_coords = coordinates[1:]
    cust_demands = demands[1:]
    
    total_demand = np.sum(cust_demands)
    # Estimate number of clusters (vehicles)
    # Add a small buffer to ensure capacity feasibility in clustering logic
    # K-means will partition space, not strictly respect capacity, but serves as spatial guide
    num_clusters = max(1, int(np.ceil(total_demand / capacity)))
    
    # Simple K-Means implementation for spatial partitioning
    # We only need cluster labels, not perfect optimization
    if num_clusters >= n - 1:
        # If K is very large, just assign each customer to its own cluster or random
        labels = np.arange(n - 1)
    else:
        # Initialize centroids randomly from customer coordinates
        np.random.seed(42) # Deterministic for reproducibility in heuristic generation
        centroids = cust_coords[np.random.choice(n - 1, num_clusters, replace=False)]
        
        labels = np.zeros(n - 1, dtype=int)
        for _ in range(10): # Fixed iterations for speed
            # Assign points to nearest centroid
            # dists[i, j] = distance between point i and centroid j
            dists = np.linalg.norm(cust_coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(num_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(cust_coords[mask], axis=0)
                else:
                    # If a cluster is empty, re-initialize it to a random point
                    new_centroids[k] = cust_coords[np.random.choice(n - 1)]
            
            # Check convergence (optional, skip for speed as 10 iters is enough)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

    # Construct Heuristic Matrix
    # H[i, j] represents attractiveness of edge i->j
    # We want to favor short edges within the same cluster and penalize long edges between clusters.
    
    # Inverse squared distance is a stronger penalty for long distances than inverse distance
    # Add epsilon to avoid division by zero
    epsilon = 1e-9
    inv_sq_dist = 1.0 / (distance_matrix + epsilon) ** 2
    
    # Initialize heuristic matrix
    H = inv_sq_dist.copy()
    
    # Apply cluster-based penalties
    # For customer-customer edges (i>0, j>0)
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            ci = labels[i - 1]
            cj = labels[j - 1]
            if ci != cj:
                # Penalize inter-cluster edges significantly
                H[i, j] *= 0.05
                
    # Ensure depot edges are reasonable
    # Depot (0) connects to all. We generally want short depot edges.
    # The inverse squared distance already handles this well.
    # No specific cluster penalty for depot edges as depot is central to all tours.
    
    # Safety checks: 
    # 1. Shape must match distance_matrix
    # 2. Values must be positive
    # 3. Values must be finite
    
    if H.shape != distance_matrix.shape:
        H = np.ones_like(distance_matrix)
        
    # Replace non-finite or non-positive values
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
# EVOLVE-BLOCK-END
