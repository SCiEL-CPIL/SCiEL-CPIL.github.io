import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Initialize weights: low degree vertices are more "valuable"
    # to keep in the independent set
    weights = 1.0 / (degrees + 1.0)
    
    # Iterative refinement: update weights based on neighbor structure
    # A vertex's value is reduced if its neighbors are also high-value
    # (they compete for selection)
    for iteration in range(20):
        if time.time() > deadline:
            break
        
        # Sum of neighbor weights
        neighbor_weight_sum = A @ weights
        
        # Average neighbor degree for normalization
        if np.all(degrees > 0):
            avg_neighbor_degree = A @ degrees / degrees
        else:
            avg_neighbor_degree = np.zeros(n)
        
        # Updated weight: base value minus penalty for having high-value neighbors
        # Normalize by average neighbor degree to better capture hub-blocking benefit
        weights_new = 1.0 / (degrees + 1.0 + 0.5 * neighbor_weight_sum / (avg_neighbor_degree + 1.0))
        
        # Check for convergence
        if np.allclose(weights, weights_new, atol=1e-10):
            weights = weights_new
            break
        weights = weights_new
    
    # Final score: negative cost of picking each vertex
    # Cost includes:
    # 1. The degree (number of edges removed)
    # 2. The sum of neighbor weights (value of eliminated vertices), normalized
    neighbor_value_sum = A @ weights
    if np.all(degrees > 0):
        avg_neighbor_degree_final = A @ degrees / degrees
    else:
        avg_neighbor_degree_final = np.zeros(n)
    
    scores = -(degrees + neighbor_value_sum / (avg_neighbor_degree_final + 1.0))
    
    return scores
