# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix (larger = more attractive) for ACO CVRP.
    
    Approach:
      1. Cluster customers into K groups using K-Means (simplified numpy version).
      2. Boost edges within the same cluster to encourage local route segments.
      3. Use inverse distance as the base attractiveness.
      4. Ensure positive values and finite results.
    
    Returns an n x n matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    deadline = start_time + 280.0

    n = distance_matrix.shape[0]
    if distance_matrix.shape[1] != n:
        return np.full((n, n), 1e-9)

    demands = np.asarray(demands, dtype=np.float64)
    if demands.shape[0] != n:
        demands = demands[:n]
    
    # Safe distance matrix
    dist_safe = np.maximum(distance_matrix, 1e-9)
    
    # Base heuristic: inverse distance
    heuristic = 1.0 / dist_safe
    
    # Structural Change: Cluster-based boosting
    # We want to encourage the ACO to visit customers in the same cluster together.
    
    # Simple K-Means implementation to avoid scipy dependency issues if any, 
    # though scipy is allowed. Let's use scipy.cluster.vq.kmeans for robustness.
    from scipy.cluster.vq import kmeans
    
    # Exclude depot (index 0) from clustering as it's the start/end point
    customer_coords = coordinates[1:]
    n_customers = n - 1
    
    if n_customers > 0:
        # Choose K based on capacity and average demand
        # Rough estimate: num_routes ~= total_demand / capacity
        total_demand = np.sum(demands[1:])
        avg_demand = total_demand / n_customers if n_customers > 0 else 1.0
        est_routes = max(1, int(total_demand / capacity))
        
        # K should be between 1 and n_customers
        # Let's try K = est_routes * 2 or so, capped
        K = min(n_customers, max(1, int(est_routes * 2)))
        
        # Run K-Means
        # Note: kmeans returns (centroids, dist)
        # We need to assign points to clusters
        try:
            centroids, _ = kmeans(customer_coords, K)
            # Assign each point to the nearest centroid
            # dist_matrix between points and centroids
            point_centroid_dists = np.linalg.norm(customer_coords[:, None, :] - centroids[None, :, :], axis=2)
            cluster_labels = np.argmin(point_centroid_dists, axis=1)
            
            # Create a cluster label array for all nodes (depot is -1 or its own cluster)
            all_labels = np.full(n, -1, dtype=int)
            all_labels[1:] = cluster_labels
            
            # Create a boolean matrix: True if i and j are in the same cluster
            # Note: Depot (0) is not in any customer cluster. 
            # Edges involving depot should not get the cluster boost, or maybe a neutral one.
            same_cluster = (all_labels[:, None] == all_labels[None, :])
            same_cluster[0, :] = False # Depot edges don't get cluster boost
            same_cluster[:, 0] = False
            
            # Boost factor for intra-cluster edges
            # Larger boost means stronger preference for local edges
            boost_factor = 5.0
            
            # Apply boost
            heuristic[same_cluster] *= boost_factor
            
            # Optional: Penalty for inter-cluster edges? 
            # Inverse distance already penalizes long distances. 
            # A penalty might be too aggressive. Let's stick to just boosting intra-cluster.
            
        except Exception:
            # If clustering fails, fall back to base heuristic
            pass

    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)

    # Final check: no NaN or Inf
    if not np.all(np.isfinite(heuristic)):
        heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
