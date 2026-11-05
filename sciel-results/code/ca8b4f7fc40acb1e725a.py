import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Compute 2-hop neighborhood sizes (local density)
    # A2 = A @ A gives number of length-2 paths between vertices
    A2 = A @ A
    # 2-hop neighbor count: for each vertex, count unique vertices at distance 2
    # Approximate by: A2[i,j] > 0 means j is at distance <= 2 from i
    # Better: count of vertices reachable in exactly 2 steps
    # For simplicity, use A2 diagonal offset as local density proxy
    # local_density[i] = number of distinct 2-hop neighbors
    # Use: (A2 > 0).sum(axis=1) - degrees[i] - 1 (exclude self and direct neighbors)
    two_hop_counts = (A2 > 0).sum(axis=1).astype(float) - degrees - 1.0
    two_hop_counts = np.maximum(two_hop_counts, 0.0)
    
    # Local clustering coefficient proxy: fraction of edges among neighbors
    # For vertex i: edges among its neighbors / (deg_i * (deg_i-1)/2)
    # Approximate using A2[i,i] which counts 2-walks from i to i = sum of degrees of neighbors
    # This is a proxy for local density
    
    # Initialize weights: combine degree and local density
    # Low degree + low local density = more valuable to keep
    local_density_score = two_hop_counts / (degrees + 1.0)
    weights = 1.0 / (degrees + 1.0 + 0.3 * local_density_score)
    
    # Iterative refinement with multiple signals
    for iteration in range(25):
        if time.time() > deadline:
            break
        
        # Sum of neighbor weights
        neighbor_weight_sum = A @ weights
        
        # Average neighbor degree for normalization
        if np.all(degrees > 0):
            avg_neighbor_degree = A @ degrees / degrees
        else:
            avg_neighbor_degree = np.zeros(n)
        
        # 2-hop neighbor weight sum (captures indirect competition)
        two_hop_weight_sum = A2 @ weights
        
        # Normalize 2-hop by typical 2-hop degree
        two_hop_deg = two_hop_counts + degrees + 1.0
        avg_two_hop_weight = two_hop_weight_sum / (two_hop_deg + 1.0)
        
        # Updated weight: 
        # - Base value inversely proportional to degree
        # - Penalty for direct high-value neighbors
        # - Smaller penalty for 2-hop neighbors (they don't directly compete)
        weights_new = 1.0 / (degrees + 1.0 + 
                             0.5 * neighbor_weight_sum / (avg_neighbor_degree + 1.0) +
                             0.15 * avg_two_hop_weight)
        
        # Check for convergence
        if np.allclose(weights, weights_new, atol=1e-10):
            weights = weights_new
            break
        weights = weights_new
    
    # Spectral component: use eigenvector centrality to identify influential vertices
    # Vertices with high centrality are "hubs" that block many others
    # We want to AVOID picking hubs if possible, so penalize high centrality
    try:
        if time.time() < deadline - 10.0:
            # Use power iteration for top eigenvector (fast, no scipy needed)
            v = np.ones(n) / np.sqrt(n)
            for _ in range(30):
                if time.time() > deadline - 5.0:
                    break
                v_new = A @ v
                norm = np.linalg.norm(v_new)
                if norm < 1e-15:
                    break
                v = v_new / norm
            # Centrality score: high = hub
            centrality = v
            # Normalize to [0, 1]
            c_min, c_max = centrality.min(), centrality.max()
            if c_max - c_min > 1e-15:
                centrality_norm = (centrality - c_min) / (c_max - c_min)
            else:
                centrality_norm = np.zeros(n)
        else:
            centrality_norm = np.zeros(n)
    except:
        centrality_norm = np.zeros(n)
    
    # Final score: negative cost of picking each vertex
    # Cost includes:
    # 1. The degree (number of edges removed)
    # 2. The sum of neighbor weights (value of eliminated vertices)
    # 3. 2-hop opportunity cost
    # 4. Spectral centrality penalty (hubs are expensive to pick)
    neighbor_value_sum = A @ weights
    two_hop_value_sum = A2 @ weights
    
    if np.all(degrees > 0):
        avg_neighbor_degree_final = A @ degrees / degrees
    else:
        avg_neighbor_degree_final = np.zeros(n)
    
    two_hop_deg_final = two_hop_counts + degrees + 1.0
    avg_two_hop_value = two_hop_value_sum / (two_hop_deg_final + 1.0)
    
    scores = -(degrees + 
               1.0 * neighbor_value_sum / (avg_neighbor_degree_final + 1.0) +
               0.3 * avg_two_hop_value +
               2.0 * centrality_norm * degrees / (degrees.mean() + 1.0))
    
    return scores
