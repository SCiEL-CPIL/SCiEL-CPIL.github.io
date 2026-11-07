import numpy as np

def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO-TSP.
    
    Uses a distance-based heuristic that favors shorter edges.
    We use (d + eps)^-alpha with alpha tuned to provide strong
    but not excessive bias toward local (short) edges.
    """
    eps = 1e-9
    d = distance_matrix
    
    # Use a moderately sharp inverse power law.
    # alpha=2.5 was the current best; try a slightly different approach:
    # combine a sharp local term with a gentle global term to avoid
    # over-concentration on a single nearest neighbor.
    #
    # Heuristic = (d + eps)^-2.5  *  (1 + (d + eps))^0.5
    # The second factor grows with distance, partially counteracting the
    # sharp decay, which gives the ants a bit more exploratory freedom
    # while still strongly preferring short edges.
    
    alpha = 2.5
    d_safe = d + eps
    base = d_safe ** (-alpha)
    # Gentle counter-weight: (1+d)^0.5 ranges from 1.0 to ~1.55 for d in [0, sqrt(2)]
    counter = np.sqrt(1.0 + d_safe)
    
    h = base * counter
    
    # Ensure all values are positive (they should be, but guard against
    # any numerical issues)
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
