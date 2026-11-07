"""Heuristic for ACO bin packing: tight-fit complementarity with structural centrality."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    
    # Compatibility matrix: fits[i,j] = True if demands[i] + demands[j] <= capacity
    fits = (demands[:, None] + demands[None, :]) <= capacity
    
    # Waste after placing both items: capacity - demands[i] - demands[j]
    waste = capacity - demands[:, None] - demands[None, :]
    
    # Degree of each item: number of items it can pair with
    degree = fits.sum(axis=1).astype(float)
    # Avoid division by zero
    degree_safe = np.maximum(degree, 1.0)
    
    # Inverse degree: constrained items get higher weight
    # An item j that is constrained (few partners) should be more attractive to place
    inv_degree_j = 1.0 / degree_safe[None, :]  # shape (1, n)
    
    # Tightness: how tight the fit is (smaller waste = better)
    # Use a smooth penalty: 1/(waste + epsilon) rewards tight fits
    eps = 1e-6
    tightness = 1.0 / (waste + eps)  # shape (n, n)
    
    # Combine: attractiveness of placing j after i
    # Weight by tightness (local fit quality) and by constraint of j (global structure)
    heuristic = tightness * inv_degree_j
    
    # For non-fitting pairs, set to very small value
    heuristic = np.where(fits, heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
