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
    for iteration in range(10):
        if time.time() > deadline:
            break
        
        # Sum of neighbor weights
        neighbor_weight_sum = A @ weights
        
        # Updated weight: base value minus penalty for having high-value neighbors
        # This captures the idea that if v has many high-value neighbors,
        # picking v is costly because it eliminates those valuable vertices
        weights_new = 1.0 / (degrees + 1.0 + 0.5 * neighbor_weight_sum)
        
        # Check for convergence
        if np.allclose(weights, weights_new, atol=1e-10):
            weights = weights_new
            break
        weights = weights_new
    
    # Final score: negative cost of picking each vertex
    # Cost includes:
    # 1. The degree (number of edges removed)
    # 2. The sum of neighbor weights (value of eliminated vertices)
    neighbor_value_sum = A @ weights
    
    scores = -(degrees + neighbor_value_sum)
    
    return scores
