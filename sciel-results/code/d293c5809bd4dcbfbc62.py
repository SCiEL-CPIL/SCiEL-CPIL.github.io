import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    degrees = adj.sum(axis=1)
    
    # Sum of neighbor degrees
    # adj @ degrees gives sum of degrees of neighbors for each vertex
    neighbor_degree_sum = adj @ degrees
    
    # Avoid division by zero
    # For vertices with degree 0, we want them to have the highest priority.
    # -degree is 0. 
    # The term neighbor_degree_sum / degree is 0/0.
    # We can use np.divide with where.
    
    # Base score: -degree (low degree is better)
    scores = -degrees
    
    # Improvement: Boost priority for vertices connected to high-degree neighbors.
    # This effectively says: "If you are a leaf attached to a hub, pick me first to disable the hub."
    # The magnitude of the boost should be proportional to the neighbor's degree.
    # alpha * (neighbor_degree_sum / degree)
    
    # If degree is 0, the term is 0. Score is 0.
    # If degree is 1, neighbor is degree D. Score is -1 + alpha*D.
    # If alpha is small, say 0.1.
    # If D=100, Score = -1 + 10 = 9.
    # A vertex with degree 10 has score -10.
    # So the leaf of the hub (9) is picked before the degree 10 vertex (-10).
    # This seems correct.
    
    # However, if a vertex has degree 50, all neighbors degree 50.
    # Score = -50 + 0.1 * 50 = -45.
    # A vertex with degree 10, neighbors degree 10.
    # Score = -10 + 0.1 * 10 = -9.
    # The degree 10 vertex is picked first.
    
    # Let's tune alpha.
    # Too large alpha might cause us to pick high-degree vertices if their neighbors are super hubs?
    # If v has degree 100, neighbors degree 100.
    # Score = -100 + 0.1 * 100 = -90.
    # Still negative.
    
    # What if alpha is too high?
    # If v has degree 1, neighbor degree 1000.
    # Score = -1 + 0.1 * 1000 = 99.
    # This vertex is picked first.
    # Is that good? Yes, it removes a hub.
    
    # Let's use a slightly more robust metric:
    # Score = -degree + alpha * mean_neighbor_degree
    
    alpha = 0.2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_deg = np.divide(neighbor_degree_sum, degrees, out=np.zeros_like(degrees), where=degrees>0)
        
    scores += alpha * mean_neighbor_deg
    
    # Check deadline
    if time.time() > deadline:
        return scores.astype(float)
        
    return scores.astype(float)
