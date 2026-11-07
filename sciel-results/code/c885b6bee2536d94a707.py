"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) or (100,) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    
    Strategy:
    1. Use the item's total prize (sum across the 5 rows if 2D, else the value).
    2. Compute per-constraint efficiency ratios: p / w_c.
    3. Combine bottleneck efficiency (min ratio) and overall efficiency (harmonic mean).
    4. Apply a power amplification to emphasize high-efficiency items.
    5. Normalize to mean 1 for stability.
    """
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape of prizes
    if prizes.ndim == 2:
        # Sum across the 5 dimensions to get a single "total prize" per item
        item_prizes = np.sum(prizes, axis=0)
    else:
        item_prizes = prizes
    
    # Ensure weights is 2D
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
        
    n_items = weights.shape[1]
    
    # Avoid division by zero
    safe_weights = np.where(weights > 1e-9, weights, 1e-9)
    
    # Ratios: (5, 100)
    ratios = item_prizes / safe_weights
    
    # Bottleneck efficiency: min ratio across constraints
    min_ratio = np.min(ratios, axis=0)
    
    # Overall efficiency: harmonic mean of ratios
    # Harmonic mean = n / sum(1/r_i)
    # Since n=5, harm_mean = 5 / sum(1/ratios, axis=0)
    inv_ratios = 1.0 / ratios
    harm_mean = 5.0 / np.sum(inv_ratios, axis=0)
    
    # Combine: emphasize bottleneck with power 1.5, and overall efficiency with power 0.5
    # This creates a score that is high only if the item is efficient in ALL constraints,
    # not just on average.
    h = (min_ratio ** 1.5) * (harm_mean ** 0.5)
    
    # Ensure positive values
    h = np.where(h > 0, h, 1e-9)
    
    # Normalize to mean 1 to keep the scale consistent with pheromone initialization
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
        
    return h
# EVOLVE-BLOCK-END
