"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices.
    Higher score means the vertex is more likely to be selected earlier by the greedy decoder.
    
    Strategy:
    1. Convert adjacency to boolean matrix.
    2. Compute degrees.
    3. For each vertex, compute the "independent set potential":
       - Low degree is good (fewer conflicts).
       - Neighbors with high degrees are good to remove (they block many others).
       - Use a composite score that balances these factors.
       
    We use a direct structural formula:
    score[i] = alpha * (mean_degree_of_neighbors[i] - degree[i]) + beta * (1 / (1 + degree[i]))
    
    This rewards vertices that:
    - Have neighbors with high average degree (removing them unblocks many)
    - Have low own degree (low conflict cost)
    """
    start_time = time.time()
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Compute degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # Compute sum of degrees of neighbors for each vertex
    # neighbor_degree_sums[i] = sum(degrees[j] for j in neighbors of i)
    neighbor_degree_sums = adj.astype(float) @ degrees
    
    # Compute mean degree of neighbors
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_neighbor_degrees = np.where(degrees > 0, neighbor_degree_sums / np.maximum(degrees, 1), 0.0)
    
    # Smooth (soft-max) transformation of neighbor degree signal:
    # Rewards vertices whose neighbors are "expensively blocked" in a smooth, bounded way.
    # Using log-sum-exp style: log(1 + exp(x)) gives a smooth max(0, x) surrogate.
    # Scale down to keep numerical stability.
    x = (mean_neighbor_degrees - degrees) / 2.0
    # softplus(x) = log(1 + exp(x)) for x > 0, ~x for large x, ~log(2) for very negative x
    softplus_x = np.where(x > 0, np.log1p(np.exp(np.minimum(x, 50))), np.log1p(np.exp(np.maximum(x, -50))))
    
    # Component 1: smooth structural benefit (softplus of the raw difference)
    component1 = softplus_x
    
    # Component 2: quadratic penalty for own degree (stronger penalty for high-degree vertices)
    # This captures that high-degree vertices block many others quadratically in expectation
    component2 = -0.1 * degrees**2 / (n + 1)
    
    # Component 3: small inverse-degree bonus for tie-breaking among similar vertices
    component3 = 0.1 / (1.0 + degrees)
    
    # Combine: structural benefit dominates, quadratic penalty shapes the tradeoff
    scores = component1 + component2 + component3
    
    # Check deadline
    if time.time() - start_time > 140:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
