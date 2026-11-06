"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign a priority score to each vertex for greedy independent set selection.
    
    Approach:
    1. Convert adjacency to a numpy array.
    2. Use a message-passing relaxation based on the fractional independent set formulation.
       The score of a vertex is inversely proportional to the sum of scores of its neighbors.
       This captures local density and degree constraints effectively, acting as a 
       spectral relaxation of the MWIS constraints.
    3. Iterate this update until convergence.
    """
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Degrees
    degrees = adj.sum(axis=1)
    
    # Initialize scores inversely proportional to degree + 1
    # This ensures low-degree vertices start with higher priority
    scores = 1.0 / (degrees + 1.0)
    
    # Message passing iterations
    # Update rule: score_i = 1 / (1 + sum_{j in N(i)} score_j)
    # This enforces a soft constraint that adjacent vertices cannot both have high scores.
    # It propagates information about local density.
    
    max_iter = 100
    for _ in range(max_iter):
        # Compute the sum of neighbor scores for each vertex
        # neighbor_sums[i] = sum_{j in N(i)} scores[j]
        neighbor_sums = adj @ scores
        
        # Update scores
        new_scores = 1.0 / (1.0 + neighbor_sums)
        
        # Check for convergence
        if np.linalg.norm(new_scores - scores) < 1e-7:
            scores = new_scores
            break
        scores = new_scores
        
    # Normalize scores to [0, 1] range for consistent scaling
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s > 1e-12:
        scores = (scores - min_s) / (max_s - min_s)
    else:
        scores = np.zeros_like(scores)
        
    return scores
# EVOLVE-BLOCK-END
