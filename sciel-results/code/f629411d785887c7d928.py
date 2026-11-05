import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    # Convert to float for operations
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Weight of a vertex u: how "valuable" is it to keep?
    # Low degree -> high value.
    # w(u) = 1 / (degree(u) + 1)
    weights = 1.0 / (degrees + 1.0)
    
    # Compute 2-hop neighbor information for better cost estimation
    # For each vertex v, the "conflict cost" includes not just its neighbors,
    # but also how those neighbors interact with other potential MIS vertices.
    
    # A @ weights gives sum of weights of direct neighbors
    neighbor_weight_sum = A @ weights
    
    # Compute clustering-like metric: for each vertex, average degree of its neighbors
    # This helps identify vertices in dense vs sparse regions
    if n > 0:
        # Average degree of neighbors (with safe division)
        neighbor_deg_sum = A @ degrees
        avg_neighbor_deg = np.where(degrees > 0, neighbor_deg_sum / degrees, 0.0)
    else:
        avg_neighbor_deg = np.zeros(n)
    
    # Cost of picking v:
    # 1. We lose v itself
    # 2. We lose all its neighbors (their weights)
    # 3. Vertices in dense regions have higher opportunity cost
    # 
    # Refined score: prioritize low-degree vertices in sparse regions
    # score = - (degree + neighbor_weight_sum + alpha * avg_neighbor_deg)
    
    # Use a small alpha to weight the clustering effect
    alpha = 0.3
    scores = -(degrees + neighbor_weight_sum + alpha * avg_neighbor_deg)
    
    # Add tiny random perturbation to break ties and avoid deterministic failures
    # on symmetric graphs. Scale is small enough not to override the main signal.
    np.random.seed(42)  # Fixed seed for reproducibility
    scores += np.random.normal(0, 1e-6, size=n)
    
    return scores
