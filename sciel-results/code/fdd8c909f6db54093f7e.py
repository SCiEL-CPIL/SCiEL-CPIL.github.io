import numpy as np

def priority(adjacency):
    # Convert to float for operations
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Avoid division by zero
    # Weight of a vertex u: how "valuable" is it to keep?
    # Low degree -> high value.
    # w(u) = 1 / (degree(u) + 1)
    weights = 1.0 / (degrees + 1.0)
    
    # Cost of picking v:
    # We lose v (cost 1, or related to its degree?)
    # And we lose its neighbors (cost is sum of their values)
    # Let's define cost as d(v) + sum(weights of neighbors)
    # Actually, the seed used -d(v). Let's stick to a similar scale.
    # If we pick v, we remove d(v) edges.
    # Let's try: Score = - (degrees + A @ weights)
    
    neighbor_value_sum = A @ weights
    
    # The score should be such that lower cost is better (higher priority)
    # So we return negative cost.
    scores = -(degrees + neighbor_value_sum)
    
    return scores
